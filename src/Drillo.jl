module Drillo

using DataFrames
using Dates
using HTTP
using JSON
using PrettyTables
using SQLite
using StatsBase

# ===========================================
#  Structs
# ===========================================

struct AppSettings
    accuracy_weight::Float64
    recency_weight::Float64
    random_noise::Float64
    epsilon::Float64
    first_quartile::Float64
    rounding_digits::Int64
end

const SETTINGS = AppSettings(0.6, 0.4, 0.15, 1e-6, 0.25, 2)

mutable struct AppState
    vocabulary::DataFrame
    activity_log::DataFrame
    max_activity_id::Int64
end

# ===========================================
#  Database Functions
# ===========================================

"""
    open_database() -> SQLite.DB

Opens and returns a connection to the vocabulary database.
"""
function open_database()
    return SQLite.DB(joinpath(@__DIR__, "..", "data", "database.db"))
end

"""
    init_vocabulary_table(db::SQLite.DB)

Creates the `vocabulary` table in the database if it doesn't exist.
"""
function init_vocabulary_table(database::SQLite.DB)
    SQLite.execute(
        database,
        """
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY,
            english TEXT,
            german TEXT,
            correct_attempts INTEGER,
            wrong_attempts INTEGER,
            last_activity TEXT,
            score REAL,
            date_added TEXT
        )
        """
    )
end

"""
    init_activity_log_table(db::SQLite.DB)

Creates the `activity_log` table in the database if it doesn't exist.
"""
function init_activity_log_table(database::SQLite.DB)
    SQLite.execute(
        database,
        """
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER,
            correct INTEGER,
            timestamp TEXT
        )
        """
    )
end

"""
    create_timestamp_index(db::SQLite.DB)

Creates an index to query SQLite database faster
"""
function create_timestamp_index(database::SQLite.DB)
    SQLite.execute(
        database,
        """
        CREATE INDEX IF NOT EXISTS idx_activity_timestamp
        ON activity_log(timestamp)
        """
    )
end

"""
    save_to_db!(df::DataFrame)

Saves the DataFrame `df` to the SQLite vocabulary table.
"""
function save_to_db!(state::AppState)
    db = open_database()
    SQLite.load!(state.vocabulary, db, "vocabulary", ifnotexists=true, replace=true)
    SQLite.load!(state.activity_log, db, "activity_log", ifnotexists=true, replace=false)
    SQLite.close(db)

    state.max_activity_id += nrow(state.activity_log)
    empty!(state.activity_log)
end

# ===========================================
#  Dataframe Functions
# ===========================================

"""
    insert_activity_log(df::DataFrame)

Appends a new activity log to the activity log DataFrame.
"""
function insert_activity_log(state::AppState, word_id::Int, correct::Bool)
    push!(state.activity_log, [nrow(state.activity_log) + state.max_activity_id + 1, word_id, correct, string(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))])
end

"""
    get_full_activity(state::AppState)::DataFrame

Concatenates historical activity data with current session's data
"""
function full_activity_log(state::AppState)::DataFrame
    db = open_database()
    historical_activity_log = DataFrame(DBInterface.execute(db, "SELECT * FROM activity_log"))
    SQLite.close(db)
    return vcat(historical_activity_log, state.activity_log)
end

"""
    init_vocabulary_dataframe(db::SQLite.DB) -> DataFrame

Loads the vocabulary table into a DataFrame.
"""
function init_vocabulary_dataframe(database::SQLite.DB)::DataFrame
    return DataFrame(DBInterface.execute(database, "SELECT * FROM vocabulary"))
end

"""
    insert_vocabulary_pair(df::Dataframe, english::String, german::String)

Appends a new word pair to the vocabulary DataFrame.
"""
function insert_vocabulary_pair(vocabulary::DataFrame, english_word::String, german_word::String)
    push!(vocabulary, [nrow(vocabulary) + 1, english_word, german_word, 0, 0, "Not tested", 0.0, string(Dates.today())])
end

"""
    update_score!(df::DataFrame, idx::Int)

Recalculates the learning score for a word based on:
- accuracy (correct vs total attempts)
- recency (days since activity, capped at 30)

Used to bias word selection in testing.

Modifies `df` in-place.
"""
function update_score!(vocabulary::DataFrame, word_idx::Int64)
    # Accuracy factor
    correct = vocabulary[word_idx, "correct_attempts"]
    wrong = vocabulary[word_idx, "wrong_attempts"]
    accuracy = correct / (correct + wrong + SETTINGS.epsilon) # To not divide by 0

    # Recency factor
    idx = vocabulary[word_idx, "last_activity"] !== "Not tested" ? vocabulary[word_idx, "last_activity"] : vocabulary[word_idx, "date_added"]
    days_since = Dates.value(Dates.today() - Dates.Date(idx))
    recency = clamp(days_since / 30, 0, 1)

    score = (1 - accuracy) * SETTINGS.accuracy_weight + recency * SETTINGS.recency_weight
    vocabulary[word_idx, "score"] = round(score, digits=SETTINGS.rounding_digits)
end

"""
    update_all_scores!(df::DataFrame)

Updates score for each row of the DataFrame by calling update_score!(idx, df) function
"""
function update_all_scores!(vocabulary::DataFrame)
    for row in 1:nrow(vocabulary)
        update_score!(vocabulary, row)
    end
end

"""
    choose_weighted_word(df::DataFrame) -> Int64

Returns an index of a row with the highest computed weight.
"""
function choose_weighted_word(vocabulary::DataFrame)::Int64
    # Compute weight: score + small randomness
    scores = vocabulary.score
    noise = rand(length(scores)) .* SETTINGS.random_noise
    weights = round.(scores .+ noise, digits=SETTINGS.rounding_digits + 1)

    # Normalize weights
    weights = max.(weights, 0.001)

    # Sample one index based on weights
    index = sample(1:nrow(vocabulary), Weights(weights))
    return index
end

# ===========================================
#  Translation Logic
# ===========================================

"""
    translate_input(text::String, trgt_lang="de", src_lang="auto") -> String

Uses Google Translate API to translate `text` into the target language.
"""
function translate_input(text_to_translate::String, target_language="de", source_language="en")::String
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=" * source_language * "&tl=" * target_language * "&dt=t&q=" * HTTP.escapeuri(text_to_translate)
    result = JSON.parse(String(HTTP.request("GET", url).body))
    join([s[1] for s in result[1]], "")
end

"""
    normalize_input_for_translation(input::String, spec::String) -> Tuple{String, Bool}

Detects if a user input is a noun based on provided specifier.
Returns a tuple of the cleaned word and a Bool indicating noun mode.
"""
function normalize_input_for_translation(raw_input::String, specifier::String)::Tuple{String,Bool}
    stripped_input = strip(raw_input) # Get rid of trailing whitespace
    noun_mode = false
    if startswith(stripped_input, specifier)
        return (stripped_input[2:end], noun_mode)
    elseif endswith(stripped_input, specifier)
        return (stripped_input[1:end-1], noun_mode)
    else
        noun_mode = true
        return (stripped_input, noun_mode)
    end
end

"""
    process_user_translation(df::DataFrame, raw_inp::String, spec::String)

Translates a word and prompts user for confirmation or correction.
"""
function process_user_translation(vocabulary::DataFrame, raw_input::String, specifier::String)
    word_to_translate, noun_mode = normalize_input_for_translation(raw_input, specifier)
    translation = noun_mode ? translate_input("the $(word_to_translate)") : lowercase.(translate_input(word_to_translate))

    println("EN: ", word_to_translate)
    println("DE: ", translation)
    print("\n✅ Accept this translation? [y/n]: ")
    confirm_translation = readline()

    if (next = check_for_mode_switch(confirm_translation)) !== nothing
        return next
    end

    if confirm_translation in ["y", "yes"] # FIX
        insert_vocabulary_pair(vocabulary, word_to_translate, translation)
    elseif confirm_translation in ["n", "no"]
        print("✏️  Enter translation for \"$(word_to_translate)\": ")
        custom_translation = readline()
        check_for_mode_switch(custom_translation)
        insert_vocabulary_pair(vocabulary, word_to_translate, custom_translation)
    else
        println("⚠️ Invalid input. Skipping...")
    end
end

# ===========================================
#  Computing
# ===========================================

"""
    compute_statistics(df:DataFrame) -> Dict

Calculates simple statistics and stores them into a Dictionary, which it returns.
"""
function compute_statistics(vocabulary::DataFrame)::Dict
    total_words = nrow(vocabulary)
    tested_words = sum(vocabulary[!, "correct_attempts"] .+ vocabulary[!, "wrong_attempts"] .> 0)
    untested_words = sum(vocabulary[!, "correct_attempts"] .+ vocabulary[!, "wrong_attempts"] .== 0)

    # Accuracy-based
    translation_accuracies = vocabulary[!, "correct_attempts"] ./ (vocabulary[!, "correct_attempts"] .+ vocabulary[!, "wrong_attempts"] .+ SETTINGS.epsilon)
    accuracy_qs = quantile(translation_accuracies, [0.25, 0.5, 0.75])
    mean_accuracy = mean(translation_accuracies)
    variance_accuracy = sum((translation_accuracies .- mean_accuracy) .^ 2) / (total_words - 1)
    std_dev_accuracy = sqrt(variance_accuracy)

    # Activity-based
    last_activity_date = [vocabulary[i, "last_activity"] == "Not tested" ? Dates.Date(vocabulary[i, "date_added"]) : Dates.Date(vocabulary[i, "last_activity"]) for i in 1:total_words]
    days_since_last_activity = Dates.value.(Dates.today() .- last_activity_date)
    mean_days_since_last_activity = mean(days_since_last_activity)
    activity_qs = quantile(days_since_last_activity, [0.25, 0.5, 0.75])
    days_to_become_stale = 10
    percentage_stale_words = sum(days_since_last_activity .> days_to_become_stale) / total_words

    return Dict(
        "Total Words" => total_words,
        "Tested Words" => tested_words,
        "Untested Words" => untested_words,
        "Average Accuracy" => round(mean_accuracy, digits=SETTINGS.rounding_digits),
        "Q1 Accuracy" => round(accuracy_qs[1], digits=SETTINGS.rounding_digits),
        "Average Recency" => round(mean_days_since_last_activity, digits=SETTINGS.rounding_digits),
        "Q3 Recency" => round(activity_qs[3], digits=SETTINGS.rounding_digits),
        "Std Dev Accuracy" => round(std_dev_accuracy, digits=SETTINGS.rounding_digits),
        "% Untested > 10 days" => round(percentage_stale_words, digits=SETTINGS.rounding_digits)
    )
end

# ===========================================
#  Mode Functions
# ===========================================

"""
    vocabulary_mode(vocab::DataFrame) -> Symbol

Runs vocabulary input loop. Returns a mode switch symbol.
"""
function vocabulary_mode(vocabulary::DataFrame)::Symbol
    println("📚 Vocabulary mode")
    println("-------------------\n")
    specifier = ""
    while true
        print("Non-noun specifier: ")
        specifier = readline()
        if (next = check_for_mode_switch(specifier)) !== nothing
            return next
        elseif length(specifier) == 1 && ispunct(specifier[1])
            break
        end
        println("\nSpecifier has to be a single punctuation character!\n")
    end
    while true
        print("\n($(specifier)) Word to translate: ")
        user_input = readline()
        if (next = check_for_mode_switch(user_input)) !== nothing
            return next
        elseif lowercase(user_input) in vocabulary[!, "english"]
            println("Duplicate. Please provide another word!")
            continue
        else
            process_user_translation(vocabulary, user_input, specifier)
        end
    end
end

"""
    test_mode(state::AppState) -> Symbol

Retrieves an english word from vocabulary table.
Prompts user for translation to german and validates it.
Updates number of attempts and activity for retrieved word and calls update_score!(df, idx) function to calculate new score.
"""
function test_mode(state::AppState)
    println("🎯 Test mode")
    println("-------------\n")
    println("# rows: $(nrow(state.vocabulary))")
    while true
        word_idx = choose_weighted_word(state.vocabulary)
        word_to_translate = state.vocabulary[word_idx, "english"]
        println("\nEN: $(word_to_translate)")
        print("DE: ")
        user_translation = readline()
        if (next = check_for_mode_switch(user_translation)) !== nothing
            return next
        else
            if user_translation == state.vocabulary[word_idx, "german"]
                state.vocabulary[word_idx, "correct_attempts"] += 1
                state.vocabulary[word_idx, "last_activity"] = string(Dates.today())
                println("\nCorrect!")
                insert_activity_log(state, word_idx, true)
            else
                state.vocabulary[word_idx, "wrong_attempts"] += 1
                state.vocabulary[word_idx, "last_activity"] = string(Dates.today())
                println("\nIncorrect!")
                insert_activity_log(state, word_idx, false)
            end
            update_score!(state.vocabulary, word_idx)
        end
    end
end

"""
    preview_mode(df::DataFrame)

Opens an HTML with current vocabulary table.
"""
function preview_mode(state::AppState)
    println("📝 Preview mode:")
    println("----------------\n")

    activity = full_activity_log(state)
    println(activity)

    stats = compute_statistics(state.vocabulary)

    # Create an in-memory buffer for the HTML output
    io = IOBuffer()

    threshold = stats["Q1 Accuracy"]
    println(threshold)

    # Highlight words with the lowest quantile in terms of translation accuracy
    hl_r = HtmlHighlighter(
        (df, i, j) -> (j == 3) && (
            df[i, 4] / (df[i, 4] + df[i, 5] + SETTINGS.epsilon) < threshold
        ),
        HtmlDecoration(color="red", font_weight="bold")
    )

    # Write the HTML output into the buffer
    pretty_table(
        io, state.vocabulary;
        backend=Val(:html),
        header=["ID", "English", "German", "✅", "❌", "Last Seen", "Score", "Added"],
        highlighters=(hl_r),
        tf=tf_html_simple,
        standalone=true
    )

    # Extract the HTML as a string
    html_output = String(take!(io))

    # Save it to a file
    html_file = "vocabulary_table.html"
    open(html_file, "w") do f
        write(f, html_output)
    end

    # Open in default browser (Windows)
    run(`cmd /c start "" "vocabulary_table.html"`)

    next = readline()
    if (next = check_for_mode_switch(next)) !== nothing
        return next
    end
end

"""
    quit_program()

Performs final cleanup and exits.
"""
function quit_program()
    println("\n❌ Program terminated!")
    exit()
end

# ===========================================
#  Mode Switching
# ===========================================

"""
    selection_mode() -> Symbol

Prompts user to select an initial mode.
"""
function selection_mode()::Symbol
    while true
        print("At any point, you can switch modes by typing:\n  🔄 [m] Mode\n  📚 [v] Vocabulary\n  🎯 [t] Test\n  📝 [p] Preview\n  ❌ [q] Quit\n> ")
        mode_selection = readline()
        if (next = check_for_mode_switch(mode_selection)) !== nothing
            return next
        else
            println("\n⚠️ Invalid mode!\n")
        end
    end
end

"""
    check_for_mode_switch(in::String) -> Symbol or nothing

Parses user input.
Returns valid mode switch or nothing.
"""
function check_for_mode_switch(input::String)::Union{Symbol,Nothing}
    key = Symbol(lowercase(strip(input)))
    return haskey(MODES, key) ? key : nothing
end

"""
    mode_loop(vocab::DataFrame, strt_mode::Symbol)

Main application loop for handling user-selected modes.
"""
function mode_loop(state::AppState, start_mode::Symbol)
    println("\nSwitching mode...\n")
    sleep(0.75)
    print("\033c")
    current_mode = start_mode
    while true
        # Call mode function, store result in next_mode (Symbol or nothing).
        next_mode = current_mode == :t || current_mode == :p ? MODES[current_mode](state) : MODES[current_mode](state.vocabulary)
        # Quit by saving DataFrames and exiting if :q is returned.
        if next_mode == :q
            MODES[:q](state)
        elseif next_mode isa Symbol
            # Switch modes, update display, and set new mode.
            println("\nSwitching mode...\n")
            sleep(0.75)
            print("\033c")
            current_mode = next_mode
        else
            println("\n⚠️ Invalid mode!\n")
        end
    end
end

# Mapping user inputs to their corresponding mode handlers
const MODES = Dict(
    :m => _ -> selection_mode(),
    :v => vocabulary_mode,
    :t => test_mode,
    :p => preview_mode,
    :q => state -> begin
        print("\033c")
        println("Saving data...")
        save_to_db!(state)
        sleep(0.5)
        println("Cleanup complete.")
        quit_program()
    end
)

# ===========================================
#  Main Entrypoint
# ===========================================

"""
    main()

Initializes the database and starts the program.
"""
function main()
    db = open_database()
    init_vocabulary_table(db)
    init_activity_log_table(db)
    atmpt_df = DataFrame(DBInterface.execute(db, "SELECT MAX(id) FROM activity_log"))
    max_id = isempty(atmpt_df) ? 0 : atmpt_df[1, 1]
    create_timestamp_index(db)
    state = AppState(
        init_vocabulary_dataframe(db),
        DataFrame(id=Int64[], word_id=Int64[], correct=Int64[], timestamp=String[]),
        max_id
    )
    println("\nWelcome to Drillo!")
    update_all_scores!(state.vocabulary)
    first = selection_mode()
    mode_loop(state, first)
end

end # module
