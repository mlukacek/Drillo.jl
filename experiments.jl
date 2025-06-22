# Testing environment

using DataFrames
using Dates
using HTTP
using JSON
using PrettyTables
using SQLite
using StatsBase

df = DataFrame(
    a = Date("2025-05-01"):Day(1):Date("2025-05-10"),
)

df.b = [x + Day(7) for x in df.a]

df

days_since_tested = [row["a"] for row in eachrow(df)]

dst = [df[i, "a"] for i in 1:nrow(df)]

dif = Dates.today() .- dst

Dates.value.(dif)

mean(1:5)

c = [10, 8, 10, 8, 8, 4]
avg_c = mean(c)

variance = sum((c .- avg_c).^2)/(length(c)-1)

stdev = sqrt(variance)

d = 2
round(10.5, d)

frame = DataFrame(
    a = collect(1:5)
)

function update(df)
    for row in eachrow(df)
        row[1] = row[1] ^ 2
    end
end

update(frame)

frame

for row in eachrow(frame)
    println(rownumber(row))
end

Dates.today()
Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

frame = DataFrame(
    a = collect(1:2:21)
)

frame[!, "b"] = frame.a .* 2
frame[!, "b"] = [val * 2 for val in frame.a]
frame[!, "b"] = [row.a * 2 for row in eachrow(frame)]

frame[!, "b"] = [val*2 for val in eachrow(frame[!, "a"])]

frame