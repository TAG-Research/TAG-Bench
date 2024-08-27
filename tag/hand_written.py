import argparse
import json
import os
import re
import time

import lotus
import pandas as pd
from lotus.models import OpenAIModel

############################################################################################################
###################################### Match based queries #################################################
############################################################################################################


def pipeline_0():
    query = "Among the schools with the average score in Math over 560 in the SAT test, how many schools are in the bay area?"
    answer = 71
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    scores_df = scores_df[scores_df["AvgScrMath"] > 560]
    scores_df = scores_df.sem_filter("{cname} is a county in the Bay Area")
    prediction = len(scores_df)
    return prediction, answer


def pipeline_1():
    query = (
        "What is the telephone number for the school with the lowest average score in reading in Southern California?"
    )
    answer = "(562) 944-0033"
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    scores_df = scores_df.sem_filter("{cname} is a county name in Southern California")
    scores_df = scores_df.loc[[scores_df["AvgScrRead"].idxmin()]]

    merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
    prediction = merged_df.Phone.values[0]
    return prediction, answer


def pipeline_3():
    query = "How many test takers are there at the school/s in a county with population over 2 million?"
    answer = 244742
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    unique_counties = pd.DataFrame(schools_df["County"].unique(), columns=["County"])
    unique_counties = unique_counties.sem_map(
        "What is the population of {County} in California? Answer with only the number without commas. Respond with your best guess."
    )
    counties_over_2m = set()
    for _, row in unique_counties.iterrows():
        try:
            if int(re.findall(r"\d+", row._map)[-1]) > 2000000:
                counties_over_2m.add(row.County)
        except:
            pass

    schools_df = schools_df[schools_df["County"].isin(counties_over_2m)]
    merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
    prediction = int(merged_df["NumTstTakr"].sum())
    return prediction, answer


def pipeline_4():
    query = "What is the grade span offered in the school with the highest longitude in cities in that are part of the 'Silicon Valley' region?"
    answer = "K-5"
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    unique_cities = pd.DataFrame(schools_df["City"].unique(), columns=["City"])
    unique_cities = unique_cities.sem_filter("{City} is a city in the Silicon Valley region")
    schools_df = schools_df[schools_df["City"].isin(unique_cities["City"])]
    schools_df = schools_df.sort_values(by=["Longitude"], key=abs, ascending=False).head(1)
    prediction = schools_df["GSoffered"].tolist()[0]
    return prediction, answer


def pipeline_5():
    query = "What are the two most common first names among the female school administrators?"
    answer = ["Jennifer", "Lisa"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")

    schools_df = (
        schools_df.groupby("AdmFName1").size().reset_index(name="count").sort_values("count", ascending=False).head(20)
    )
    schools_df = schools_df.sem_filter("{AdmFName1} is a female first name")
    prediction = schools_df["AdmFName1"].tolist()[:2]
    return prediction, answer


def pipeline_6():
    query = "Among the posts owned by csgillespie, how many of them are root posts and mention academic papers?"
    answer = 4
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = users_df[users_df["DisplayName"] == "csgillespie"]
    posts_df = posts_df[posts_df["ParentId"].isna()]
    merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId")
    merged_df = merged_df.sem_filter("{Body} mentions academic papers")

    prediction = len(merged_df)
    return prediction, answer


def pipeline_8():
    query = "How many of the comments with a score of 17 are about statistics?"
    answer = 4
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    comments_df = comments_df[comments_df["Score"] == 17]
    comments_df = comments_df.sem_filter("{Text} is about statistics")
    prediction = len(comments_df)

    return prediction, answer


def pipeline_10():
    query = "Of the posts with views above 80000, how many discuss the R programming language?"
    answer = 3
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    posts_df = posts_df[posts_df["ViewCount"] > 80000]
    posts_df = posts_df.sem_filter("{Body} discusses the R programming language")
    prediction = len(posts_df)

    return prediction, answer


def pipeline_11():
    query = "Please give the names of the races held on the circuits in the middle east."
    answer = [
        "Bahrain Grand Prix",
        "Turkish Grand Prix",
        "Abu Dhabi Grand Prix",
        "Azerbaijan Grand Prix",
        "European Grand Prix",
    ]
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    circuits_df = circuits_df.sem_filter("{country} is in the Middle East")

    merged_df = pd.merge(circuits_df, races_df, on="circuitId", suffixes=["_circuit", "_race"]).drop_duplicates(
        subset="name_race"
    )
    prediction = merged_df["name_race"].tolist()

    return prediction, answer


def pipeline_13():
    query = "How many Asian drivers competed in the 2008 Australian Grand Prix?"
    answer = 2

    drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
    drivers_df = drivers_df.sem_filter("{nationality} is Asian")
    races_df = races_df[(races_df["name"] == "Australian Grand Prix") & (races_df["year"] == 2008)]
    merged_df = pd.merge(pd.merge(races_df, results_df, on="raceId"), drivers_df, on="driverId")
    prediction = len(merged_df)

    return prediction, answer


def pipeline_16():
    query = "What is the preferred foot when attacking of the player with the most Ballon d'Or awards of all time?"
    answer = "left"
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
    key_player = lm(
        [
            {
                "role": "user",
                "content": "What is the first and last name of the player who has won the most Ballon d'Or awards of all time? Respond with only the name and no other words.",
            }
        ]
    )[0]
    players_df = players_df[players_df["player_name"] == key_player]
    merged_df = pd.merge(players_df, attributes_df, on="player_api_id")
    merged_df = merged_df[["player_name", "preferred_foot"]]
    prediction = merged_df["preferred_foot"].values[0]
    return prediction, answer


def pipeline_18():
    query = "List the football player with a birthyear of 1970 who is an Aquarius"
    answer = "Hans Vonk"
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    players_df = players_df[players_df["birthday"].str.startswith("1970")]
    players_df = players_df.sem_filter("Someone born on {birthday} would be an Aquarius")
    prediction = players_df["player_name"].values[0]
    return prediction, answer


def pipeline_19():
    query = "Please list the league from the country which is landlocked."
    answer = "Switzerland Super League"
    leagues_df = pd.read_csv("../pandas_dfs/european_football_2/League.csv")
    countries_df = pd.read_csv("../pandas_dfs/european_football_2/Country.csv")
    countries_df = countries_df.sem_filter("{name} is landlocked")
    merged_df = pd.merge(
        leagues_df, countries_df, left_on="country_id", right_on="id", suffixes=["_league", "_country"]
    )
    prediction = merged_df["name_league"].values[0]

    return prediction, answer


def pipeline_20():
    query = "How many matches in the 2008/2009 season were held in countries where French is an official language?"
    answer = 866
    matches_df = pd.read_csv("../pandas_dfs/european_football_2/Match.csv")
    countries_df = pd.read_csv("../pandas_dfs/european_football_2/Country.csv")
    matches_df = matches_df[matches_df["season"] == "2008/2009"]
    countries_df = countries_df.sem_filter("{name} has French as an official language")
    merged_df = pd.merge(matches_df, countries_df, left_on="country_id", right_on="id")
    prediction = len(merged_df)

    return prediction, answer


def pipeline_21():
    query = "Of the top three away teams that scored the most goals, which one has the most fans?"
    answer = "FC Barcelona"
    teams_df = pd.read_csv("../pandas_dfs/european_football_2/Team.csv")
    matches_df = pd.read_csv("../pandas_dfs/european_football_2/Match.csv")

    merged_df = pd.merge(matches_df, teams_df, left_on="away_team_api_id", right_on="team_api_id")
    merged_df = (
        merged_df.sort_values("away_team_goal", ascending=False).drop_duplicates(subset="team_long_name").head(3)
    )
    prediction = merged_df.sem_topk("What {team_long_name} has the most fans?", 1).team_long_name.values[0]
    return prediction, answer


def pipeline_24():
    query = "Which year recorded the most gas use paid in the higher value currency?"
    answer = 2013
    customers_df = pd.read_csv("../pandas_dfs/debit_card_specializing/customers.csv")
    yearmonth_df = pd.read_csv("../pandas_dfs/debit_card_specializing/yearmonth.csv")

    unique_currencies = customers_df["Currency"].unique()
    most_value = (
        pd.DataFrame(unique_currencies, columns=["Currency"])
        .sem_topk("What {Currency} is the highest value currency?", 1)
        .Currency.values[0]
    )
    customers_df = customers_df[customers_df["Currency"] == most_value]

    yearmonth_df["year"] = yearmonth_df["Date"] // 100
    merged_df = pd.merge(customers_df, yearmonth_df, on="CustomerID")
    merged_df = merged_df.groupby("year")["Consumption"].sum().reset_index()
    merged_df = merged_df.sort_values("Consumption", ascending=False)
    prediction = int(merged_df["year"].values[0])

    return prediction, answer


def pipeline_108():
    query = "Among the posts that were voted by user 1465, determine if the post is relevant to machine learning. Respond with YES if it is and NO if it is not."
    answer = ["YES", "YES", "YES"]
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    votes_df = pd.read_csv("../pandas_dfs/codebase_community/votes.csv")
    votes_df = votes_df[votes_df["UserId"] == 1465]
    merged_df = pd.merge(posts_df, votes_df, left_on="Id", right_on="PostId")
    merged_df = merged_df.sem_map(
        "{Body} is relevant to machine learning. Answer with YES if it is and NO if it is not."
    )
    prediction = merged_df._map.tolist()
    return prediction, answer


def pipeline_109():
    query = "Extract the statistical term from the post titles which were edited by Vebjorn Ljosa."
    answer = ["beta-binomial distribution", "AdaBoost", "SVM", "Kolmogorov-Smirnov statistic"]
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    merged_df = pd.merge(posts_df, users_df, left_on="OwnerUserId", right_on="Id")
    merged_df = merged_df[merged_df["DisplayName"] == "Vebjorn Ljosa"]
    merged_df = merged_df.sem_map("Extract the statistical term from {Title}. Respond with only the statistical term.")
    prediction = merged_df._map.tolist()
    return prediction, answer


def pipeline_110():
    query = "List the Comment Ids of the positive comments made by the top 5 newest users on the post with the title 'Analysing wind data with R'"
    answer = [11449]
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    posts_df = posts_df[posts_df["Title"] == "Analysing wind data with R"]
    merged_df = pd.merge(comments_df, posts_df, left_on="PostId", right_on="Id", suffixes=["_comment", "_post"])
    merged_df = pd.merge(merged_df, users_df, left_on="UserId", right_on="Id", suffixes=["_merged", "_user"])
    merged_df = merged_df.sort_values(by=["CreationDate_user"], ascending=False).head(5)
    merged_df = merged_df.sem_filter("The sentiment of {Text} is positive")
    prediction = merged_df.Id_comment.tolist()
    return prediction, answer


def pipeline_111():
    query = 'For the post from which the tag "bayesian" is excerpted from, identify whether the body of the post is True or False. Answer with True or False ONLY.'
    answer = "TRUE"
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    tags_df = pd.read_csv("../pandas_dfs/codebase_community/tags.csv")
    tags_df = tags_df[tags_df["TagName"] == "bayesian"]
    merged_df = pd.merge(tags_df, posts_df, left_on="ExcerptPostId", right_on="Id")
    prediction = merged_df.sem_map("Determine whether the content in {Body} is true. Respond with only TRUE or FALSE.")[
        "_map"
    ].values[0]
    return prediction, answer


############################################################################################################
###################################### Comparison based queries ############################################
############################################################################################################
def pipeline_25():
    query = "What is the average total price of the transactions taken place in gas stations in the country which is historically known as Bohemia, to the nearest integer?"
    answer = "453"
    transactions_df = pd.read_csv("../pandas_dfs/debit_card_specializing/transactions_1k.csv")
    gasstations_df = pd.read_csv("../pandas_dfs/debit_card_specializing/gasstations.csv")

    countries = lm(
        [
            {
                "role": "user",
                "content": "What are abbreviations for the country historically known as Bohemia? If there are multiple possible abbreivations list them as a python list with quotes around each abbreviation.",
            }
        ]
    )[0]
    try:
        countries = eval(countries)
    except:
        countries = [countries]

    gasstations_df = gasstations_df[gasstations_df["Country"].isin(countries)]

    merged_df = pd.merge(transactions_df, gasstations_df, on="GasStationID")
    prediction = round(merged_df["Price"].mean())

    return prediction, answer


def pipeline_27():
    query = (
        "List the username of the oldest user located in the capital city of Austria who obtained the Supporter badge."
    )
    answer = "ymihere"
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    badges_df = pd.read_csv("../pandas_dfs/codebase_community/badges.csv")
    merged_df = pd.merge(users_df, badges_df, left_on="Id", right_on="UserId")
    merged_df = merged_df[merged_df["Name"] == "Supporter"]

    location = lm(
        [
            {
                "role": "user",
                "content": "What is the capital city of Austria? Respond with only the city name and no other words.",
            }
        ]
    )[0]
    location = f"{location}, Austria"
    merged_df = merged_df[merged_df["Location"] == location]
    prediction = merged_df.sort_values(by=["Age"], ascending=False).DisplayName.values.tolist()[0]

    return prediction, answer


def pipeline_29():
    query = "What is the difference in gas consumption between customers who pay using the currency of the Czech Republic and who pay the currency of European Union in 2012, to the nearest integer?"
    answer = 402524570
    customers_df = pd.read_csv("../pandas_dfs/debit_card_specializing/customers.csv")
    yearmonth_df = pd.read_csv("../pandas_dfs/debit_card_specializing/yearmonth.csv")

    countries = {"Area": ["Czech Republic", "European Union"]}
    countries_df = pd.DataFrame(countries)
    currency_df = countries_df.sem_map(
        "Given {Area}, return the 3 letter currency code for the area. Answer with the code ONLY.", suffix="currency"
    )
    currencies = currency_df["currency"].values.tolist()

    yearmonth_df = yearmonth_df[yearmonth_df["Date"] // 100 == 2012]

    merged_df = pd.merge(customers_df, yearmonth_df, on="CustomerID")
    first_df = merged_df[merged_df["Currency"] == currencies[0]]
    second_df = merged_df[merged_df["Currency"] == currencies[1]]

    prediction = round(first_df["Consumption"].sum() - second_df["Consumption"].sum())

    return prediction, answer


def pipeline_30():
    query = "Is it true that more SMEs pay in Czech koruna than in the second-largest reserved currency in the world?"
    answer = "Yes"
    customers_df = pd.read_csv("../pandas_dfs/debit_card_specializing/customers.csv")

    customers_df = customers_df[customers_df["Segment"] == "SME"]

    first_df = customers_df[customers_df["Currency"] == "CZK"]

    currency = lm(
        [
            {
                "role": "user",
                "content": "What is the 3 letter code for the second-largest reserved currency in the world? Respond with only the 3 letter code and no other words.",
            }
        ]
    )[0]
    second_df = customers_df[customers_df["Currency"] == currency]

    if len(first_df) > len(second_df):
        prediction = "Yes"
    else:
        prediction = "No"

    return prediction, answer


def pipeline_33():
    query = "What is the total number of schools whose total SAT scores are greater or equal to 1500 whose mailing city is the county seat of Lake County, California?"
    answer = 2
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")

    city = lm(
        [
            {
                "role": "user",
                "content": "What is the name of the city that is the county seat of Lake County, California? Respond with only the city name and no other words.",
            }
        ]
    )[0]
    schools_df = schools_df[schools_df["City"] == city]
    scores_df["total"] = scores_df["AvgScrRead"] + scores_df["AvgScrMath"] + scores_df["AvgScrWrite"]
    scores_df = scores_df[scores_df["total"] >= 1500]
    merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
    prediction = len(merged_df)

    return prediction, answer


def pipeline_35():
    query = "How many drivers born after the end of the Vietnam War have been ranked 2?"
    answer = 27
    drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
    year_df = drivers_df.head(1).sem_map(
        "Given {driverId}, return the year of the end of the Vietnam War. Answer with the year ONLY.", suffix="year"
    )
    vietnamyear = int(year_df["year"].values.tolist()[0])
    drivers_df["birthyear"] = drivers_df.dropna(subset=["dob"])["dob"].str[:4].astype(int)
    drivers_df = drivers_df[drivers_df["birthyear"] > vietnamyear]
    prediction = len(drivers_df)

    return prediction, answer


def pipeline_36():
    query = "Among all European Grand Prix races, what is the percentage of the races were hosted in the country where the Bundesliga happens, to the nearest whole number?"
    answer = 52
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    races_df = races_df[races_df["name"] == "European Grand Prix"]
    merged_df = pd.merge(circuits_df, races_df, on="circuitId")
    denom = len(merged_df)
    merged_df = merged_df.sem_filter("{country} is where the Bundesliga happens")
    numer = len(merged_df)
    prediction = int(numer * 100 / denom)
    return prediction, answer


def pipeline_37():
    query = "From 2010 to 2015, what was the average overall rating, rounded to the nearest integer, of players who are higher than 170 and shorter than Michael Jordan?"
    answer = 69
    jordan_df = pd.DataFrame({"Name": ["Michael Jordan"]})
    jordan_df = jordan_df.sem_map(
        "Given {Name}, provide the height in cm. Answer with ONLY the number to one decimal place.", suffix="height"
    )
    jordan_height = float(jordan_df["height"].values.tolist()[0])

    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    players_df = players_df[players_df["height"] > 170]
    players_df = players_df[players_df["height"] < jordan_height]

    attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
    attributes_df["year"] = attributes_df["date"].str[:4].astype(int)
    attributes_df = attributes_df[attributes_df["year"] >= 2010]
    attributes_df = attributes_df[attributes_df["year"] <= 2015]

    merged_df = pd.merge(players_df, attributes_df, on="player_api_id")
    prediction = round(merged_df["overall_rating"].mean())

    return prediction, answer


def pipeline_38():
    query = "Among the drivers that finished the race in the 2008 Australian Grand Prix, how many debuted earlier than Lewis Hamilton?"
    answer = 3
    drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")

    races_df = races_df[races_df["name"] == "Australian Grand Prix"]
    races_df = races_df[races_df["year"] == 2008]
    results_df = results_df[results_df["time"].notnull()]
    drivers_df = drivers_df.sem_map(
        "What year did driver {forename} {surname} debut in Formula 1? Answer with the year ONLY.", suffix="debut"
    )
    drivers_df["debut"] = pd.to_numeric(drivers_df["debut"], errors="coerce")
    drivers_df = drivers_df.dropna(subset=["debut"])

    merged_df = pd.merge(results_df, races_df, on="raceId").merge(drivers_df, on="driverId")
    merged_df = merged_df[merged_df["year"] > merged_df["debut"]]

    prediction = len(merged_df)

    return prediction, answer


def pipeline_39():
    query = "How many players were born after the year of the 14th FIFA World Cup?"
    answer = 3028
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    year_df = players_df.head(1).sem_map(
        "Given {id}, return the year of the 14th FIFA World Cup. Answer with the year ONLY.", suffix="year"
    )
    wcyear = int(year_df["year"].values.tolist()[0])
    players_df["birthyear"] = players_df["birthday"].str[:4].astype(int)
    players_df = players_df[players_df["birthyear"] > wcyear]
    prediction = len(players_df)

    return prediction, answer


def pipeline_40():
    query = "Among the players whose height is over 180, how many of them have a volley score of over 70 and are taller than Stephen Curry?"
    answer = 88
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
    players_df = players_df[players_df["height"] >= 180]
    attributes_df = attributes_df[attributes_df["volleys"] > 70]
    merged_df = pd.merge(players_df, attributes_df, on="player_api_id").drop_duplicates(subset="player_api_id")

    steph_height = lm(
        [
            {
                "role": "user",
                "content": "How tall is Stephen Curry in centimeters? Box your final answer with \\boxed{just a number}.",
            }
        ]
    )[0]
    steph_height = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", steph_height).group(1))
    merged_df = merged_df[merged_df["height"] > int(steph_height)]
    prediction = len(merged_df)

    return prediction, answer


def pipeline_41():
    query = "Give the number of schools with the percent eligible for free meals in K-12 is more than 0.1 and test takers whose test score is greater than or equal to the score one hundred points less than the maximum."
    answer = 1
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
    frpm_df = frpm_df[(frpm_df["Free Meal Count (K-12)"] / frpm_df["Enrollment (K-12)"]) > 0.1]
    max_score = lm(
        [
            {
                "role": "user",
                "content": "What is the maximum SAT score?. Box your final answer with \\boxed{just a number}.",
            }
        ]
    )[0]
    max_score = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", max_score).group(1))
    scores_df = scores_df[scores_df["AvgScrRead"] + scores_df["AvgScrMath"] >= max_score - 300]
    merged_df = pd.merge(scores_df, frpm_df, left_on="cds", right_on="CDSCode").drop_duplicates(subset="cds")
    prediction = len(merged_df)

    return prediction, answer


def pipeline_42():
    query = "How many schools have the difference in enrollements between K-12 and ages 5-17 as more than average high school class size?"
    answer = 1236
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
    avg_class_size = lm(
        [
            {
                "role": "user",
                "content": "What is the average high school class size? Box your final answer with \\boxed{just a number}.",
            }
        ]
    )[0]
    avg_class_size = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", avg_class_size).group(1))
    frpm_df = frpm_df[frpm_df["Enrollment (K-12)"] - frpm_df["Enrollment (Ages 5-17)"] > avg_class_size]
    merged_df = pd.merge(schools_df, frpm_df, on="CDSCode").drop_duplicates(subset="CDSCode")
    prediction = len(merged_df)

    return prediction, answer


def pipeline_43():
    query = "Among the users who have more than 100 upvotes, how many of them are older than the median age in America?"
    answer = 32
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    users_df = users_df[users_df["UpVotes"] > 100]
    median_age = lm(
        [
            {
                "role": "user",
                "content": "What is the median age in America? Box your final answer with \\boxed{just a number}.",
            }
        ]
    )[0]
    median_age = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", median_age).group(1))
    users_df = users_df[users_df["Age"] > median_age]
    prediction = len(users_df)

    return prediction, answer


def pipeline_44():
    query = "Please list the player names taller than 6 foot 8?"
    answer = ["Kristof van Hout"]
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    height = lm(
        [
            {
                "role": "user",
                "content": "What is 6 foot 8 in centimeters? Box your final answer with \\boxed{just a number}.",
            }
        ]
    )[0]
    height = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", height).group(1))
    players_df = players_df[players_df["height"] > height]
    prediction = players_df["player_name"].values.tolist()
    return prediction, answer


def pipeline_45():
    query = "How many players whose first names are Adam and weigh more than 77.1kg?"
    answer = 24
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    players_df = players_df[players_df["player_name"].str.startswith("Adam")]
    pounds = lm(
        [{"role": "user", "content": "What is 77.1kg in pounds? Box your final answer with \\boxed{just a number}."}]
    )[0]
    pounds = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", pounds).group(1))
    players_df = players_df[players_df["weight"] > pounds]
    prediction = len(players_df)

    return prediction, answer


def pipeline_46():
    query = "Please provide the names of top three football players who are over 5 foot 11 tall in alphabetical order."
    answer = ["Aaron Appindangoye", "Aaron Galindo", "Aaron Hughes"]
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    height = lm(
        [
            {
                "role": "user",
                "content": "What is 5 foot 11 in centimeters? Box your final answer with \\boxed{just a number}.",
            }
        ]
    )[0]
    height = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", height).group(1))
    players_df = players_df[players_df["height"] > height]
    players_df = players_df.sort_values("player_name")
    prediction = players_df["player_name"].head(3).values.tolist()

    return prediction, answer


def pipeline_47():
    query = "How many transactions taken place in the gas station in the Czech Republic are with a price of over 42.74 US dollars?"
    answer = 56
    transactions_df = pd.read_csv("../pandas_dfs/debit_card_specializing/transactions_1k.csv")
    gasstations_df = pd.read_csv("../pandas_dfs/debit_card_specializing/gasstations.csv")
    gasstations_df = gasstations_df[gasstations_df["Country"] == "CZE"]
    merged_df = pd.merge(transactions_df, gasstations_df, on="GasStationID")

    price = lm(
        [{"role": "user", "content": "What is 42.74 USD in CZK? Box your final answer with \\boxed{just a number}."}]
    )[0]
    price = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", price).group(1))
    merged_df = merged_df[merged_df["Price"] > price]
    prediction = len(merged_df)

    return prediction, answer


def pipeline_48():
    query = "Which of these circuits is located closer to a capital city, Silverstone Circuit, Hockenheimring or Hungaroring?"
    answer = "Hungaroring"
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    circuits_df = circuits_df[circuits_df["name"].isin(["Silverstone Circuit", "Hockenheimring", "Hungaroring"])]

    prediction = circuits_df.sem_topk("What circuit, named {name} is located closer to a capital city?", 1).name.values[
        0
    ]

    return prediction, answer


def pipeline_49():
    query = "Which race was Alex Yoong in when he was in the top half of finishers?"
    answer = "Australian Grand Prix"
    drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
    drivers_df = drivers_df[(drivers_df["forename"] == "Alex") & (drivers_df["surname"] == "Yoong")]
    driverStandings_df = pd.read_csv("../pandas_dfs/formula_1/driverStandings.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    merged_df = pd.merge(drivers_df, driverStandings_df, on="driverId").merge(races_df, on="raceId")

    prediction = merged_df.sem_filter(
        "The {position} is in the top half of number racers in a formula 1 race."
    ).name.values[0]

    return prediction, answer


############################################################################################################
###################################### Ranking based queries ###############################################
############################################################################################################


def pipeline_50():
    query = "Among the magnet schools with SAT test takers of over 500, which school name sounds most futuristic?"
    answer = "Polytechnic High"

    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    schools_df = schools_df[schools_df["Magnet"] == 1]
    satscores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    satscores_df = satscores_df[satscores_df["NumTstTakr"] > 500]
    merged_df = pd.merge(schools_df, satscores_df, left_on="CDSCode", right_on="cds")
    prediction = merged_df.sem_topk("What {School} sounds most futuristic?", 1).School.values[0]

    return prediction, answer


def pipeline_51():
    query = "Of the 5 posts wih highest popularity, list their titles in order of most technical to least technical."
    answer = [
        "How to interpret and report eta squared / partial eta squared in statistically significant and non-significant analyses?",
        "How to interpret F- and p-value in ANOVA?",
        "What is the meaning of p values and t values in statistical tests?",
        "How to choose between Pearson and Spearman correlation?",
        "How do I get the number of rows of a data.frame in R?",
    ]

    posts_df = (
        pd.read_csv("../pandas_dfs/codebase_community/posts.csv").sort_values(by=["ViewCount"], ascending=False).head(5)
    )

    prediction = posts_df.sem_topk("What {Title} is most technical?", 5).Title.values.tolist()

    return prediction, answer


def pipeline_52():
    query = "What are the Post Ids of the top 2 posts in order of most grateful comments received on 9-14-2014"
    answer = [115372, 115254]

    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    comments_df = comments_df[comments_df["CreationDate"].str.startswith("2014-09-14")]
    merged_df = pd.merge(posts_df, comments_df, left_on="Id", right_on="PostId")
    merged_df = merged_df.sem_filter("The sentiment on {Text} is that of someone being grateful.")
    merged_df = merged_df.groupby("Id_x").size().sort_values(ascending=False)
    prediction = list(merged_df.index[:2])

    return prediction, answer


def pipeline_53():
    query = "For the post owned by csgillespie with the highest popularity, what is the most sarcastic comment?"
    answer = "That pirates / global warming chart is clearly cooked up by conspiracy theorists - anyone can see they have deliberately plotted even spacing for unequal time periods to avoid showing the recent sharp increase in temperature as pirates are almost entirely wiped out. We all know that as temperatures rise it makes the rum evaporate and pirates cannot survive those conditions."

    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    users_df = users_df[users_df["DisplayName"] == "csgillespie"]
    merged_df = (
        pd.merge(posts_df, users_df, left_on="OwnerUserId", right_on="Id")
        .sort_values(by=["ViewCount"], ascending=False)
        .head(1)
    )

    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    merged_df_with_comments = pd.merge(merged_df, comments_df, left_on="Id_x", right_on="PostId")
    prediction = merged_df_with_comments.sem_topk("What {Text} is most sarcastic?", 1).Text.values[0]

    return prediction, answer


def pipeline_54():
    query = "Among the top 10 most popular tags, which is the least related to statistics?"
    answer = "self-study"
    tags_df = pd.read_csv("../pandas_dfs/codebase_community/tags.csv")
    tags_df = tags_df.sort_values("Count", ascending=False)
    tags_df = tags_df.head(10)
    prediction = tags_df.sem_topk("{TagName} is the least related to statistics?", 1).TagName.values[0]
    return prediction, answer


def pipeline_55():
    query = "Of the top 10 most favorited posts, what is the Id of the most lighthearted post?"
    answer = 423
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    posts_df = posts_df.sort_values("FavoriteCount", ascending=False).head(10)
    prediction = posts_df.sem_topk("What {Body} is most lighthearted?", 1).Id.values[0]
    prediction = int(prediction)

    return prediction, answer


def pipeline_56():
    query = "Among the posts owned by a user over 65 with a score of over 10, what are the post id's of the top 2 posts made with the least expertise?"
    answer = [8485, 15670]
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    users_df = users_df[users_df["Age"] > 65]
    posts_df = posts_df[posts_df["Score"] > 10]
    merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId", suffixes=["_users", "_posts"])
    prediction = merged_df.sem_topk("What {Body} is made with the least expertise?", 2).Id_posts.values.tolist()

    return prediction, answer


def pipeline_57():
    query = "Among the badges obtained by csgillespie in 2011, which is the most creatively named?"
    answer = "Strunk & White"
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    badges_df = pd.read_csv("../pandas_dfs/codebase_community/badges.csv")
    users_df = users_df[users_df["DisplayName"] == "csgillespie"]
    merged_df = pd.merge(users_df, badges_df, left_on="Id", right_on="UserId").drop_duplicates("Name")
    prediction = merged_df.sem_topk("What {Name} is most creative?", 1).Name.values[0]

    return prediction, answer


def pipeline_58():
    query = "Of the posts owned by Yevgeny, what are the id's of the top 3 most pessimistic?"
    answer = [23819, 24216, 35748]
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = users_df[users_df["DisplayName"] == "Yevgeny"]
    merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId", suffixes=["_users", "_posts"])
    prediction = merged_df.sem_topk("What {Body} is most pessimistic?", 3).Id_posts.values.tolist()

    return prediction, answer


def pipeline_59():
    query = "Of the top 10 players taller than 180 ordered by average heading accuracy, what are the top 3 most unique sounding names?"
    answer = ["Naldo", "Per Mertesacker", "Didier Drogba"]
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
    players_df = players_df[players_df["height"] > 180]
    merged_df = pd.merge(players_df, attributes_df, on="player_api_id", suffixes=["_players", "_attributes"])
    merged_df = merged_df.groupby("player_api_id")
    merged_df = merged_df.agg({"heading_accuracy": "mean", "player_name": "first"}).reset_index()
    merged_df = merged_df.sort_values("heading_accuracy", ascending=False).head(10)
    prediction = merged_df.sem_topk("What {player_name} is most unique sounding?", 3).player_name.values.tolist()

    return prediction, answer


def pipeline_60():
    query = "Out of users that have obtained at least 200 badges, what are the top 2 display names that seem most based off a real name?"
    answer = ["Jeromy Anglim", "Glen_b"]
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    badges_df = pd.read_csv("../pandas_dfs/codebase_community/badges.csv")
    merged_df = pd.merge(users_df, badges_df, left_on="Id", right_on="UserId")
    merged_df = merged_df.groupby("DisplayName").filter(lambda x: len(x) >= 200).drop_duplicates(subset="DisplayName")
    prediction = merged_df.sem_topk(
        "What {DisplayName} seems most based off a real name?", 2
    ).DisplayName.values.tolist()
    return prediction, answer


def pipeline_61():
    query = "Of the cities containing exclusively virtual schools which are the top 3 safest places to live?"
    answer = ["Thousand Oaks", "Simi Valley", "Westlake Village"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    schools_df = schools_df[schools_df["Virtual"] == "F"].drop_duplicates(subset="City")
    prediction = schools_df.sem_topk("What {City} is the safest place to live?", 3).City.values.tolist()
    return prediction, answer


def pipeline_62():
    query = "List the cities containing the top 5 most enrolled schools in order from most diverse to least diverse. "
    answer = ["Long Beach", "Paramount", "Granada Hills", "Temecula", "Carmichael"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
    merged_df = pd.merge(schools_df, frpm_df, on="CDSCode")
    merged_df = merged_df.sort_values("Enrollment (K-12)", ascending=False).head(5)
    prediction = merged_df.sem_topk("What {City} is the most diverse?", 5).City.values.tolist()
    return prediction, answer


def pipeline_63():
    query = "Please list the top three continuation schools with the lowest eligible free rates for students aged 5-17 and rank them based on the overall affordability of their respective cities."
    answer = ["Del Amigo High (Continuation)", "Rancho del Mar High (Continuation)", "Millennium High Alternative"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
    frpm_df = frpm_df[frpm_df["Educational Option Type"] == "Continuation School"]
    frpm_df["frpm_rate"] = frpm_df["Free Meal Count (Ages 5-17)"] / frpm_df["Enrollment (Ages 5-17)"]
    frpm_df = frpm_df.sort_values("frpm_rate", ascending=True).head(3)
    merged_df = pd.merge(schools_df, frpm_df, on="CDSCode")
    prediction = merged_df.sem_topk("What {City} is the most affordable?", 3).School.values.tolist()
    return prediction, answer


def pipeline_64():
    query = "Of the schools with the top 3 SAT excellence rate, order their counties by academic reputation from strongest to weakest."
    answer = "Santa Clara County"
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    satscores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    satscores_df["excellence_rate"] = satscores_df["NumGE1500"] / satscores_df["NumTstTakr"]
    satscores_df = satscores_df.sort_values("excellence_rate", ascending=False).head(3)
    merged_df = pd.merge(schools_df, satscores_df, left_on="CDSCode", right_on="cds")
    prediction = merged_df.sem_topk("What {County} has the strongest academic reputation?", 3).County.values[0]
    return prediction, answer


def pipeline_65():
    query = "Among the cities with the top 10 lowest enrollment for students in grades 1 through 12, which are the top 2 most popular cities to visit?"
    answer = ["Death Valley", "Shaver Lake"]
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
    merged_df = pd.merge(schools_df, frpm_df, on="CDSCode")
    merged_df = merged_df.groupby("City").agg({"Enrollment (K-12)": "sum"}).reset_index()
    merged_df = merged_df.sort_values("Enrollment (K-12)", ascending=True).head(10)
    prediction = merged_df.sem_topk(
        "What {City}/location in California is the most popular to visit?", 2
    ).City.values.tolist()
    return prediction, answer


def pipeline_106():
    query = "Of the top 5 users with the most views, who has their social media linked in their AboutMe section?"
    answer = "whuber"
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    users_df = users_df.sort_values("Views", ascending=False).head(5)
    prediction = users_df.sem_filter("The {AboutMe} contains a link to social media.").DisplayName.values[0]
    return prediction, answer


def pipeline_107():
    query = "Of all the comments commented by the user with a username of Harvey Motulsky and with a score of 5, rank the post ids in order of most helpful to least helpful."
    answer = [89457, 64710, 4945]
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    users_df = users_df[users_df["DisplayName"] == "Harvey Motulsky"]
    comments_df = comments_df[comments_df["Score"] == 5]
    merged_df = pd.merge(users_df, comments_df, left_on="Id", right_on="UserId")
    prediction = merged_df.sem_topk("What {Text} is most helpful?", 3).PostId.values.tolist()
    return prediction, answer


def pipeline_952():
    query = "Of the constructors that have been ranked 1 in 2014, which has the most prestige"
    answer = "Ferrari"
    constructors_df = pd.read_csv("../pandas_dfs/formula_1/constructors.csv")
    results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    merged_df = pd.merge(results_df, constructors_df, on="constructorId", suffixes=["_results", "_constructors"])
    merged_df = pd.merge(merged_df, races_df, on="raceId", suffixes=["_merged", "_races"])
    merged_df = merged_df[(merged_df["rank"] == 1) & (merged_df["year"] == 2014)].drop_duplicates(
        subset="constructorId"
    )
    merged_df = merged_df.rename(columns={"name_merged": "name"})
    prediction = merged_df.sem_topk("What {name} is most prestigious?", 1).name.values[0]
    return prediction, answer


def pipeline_1000():
    query = "Of the 5 racetracks that hosted the most recent races, rank the locations by distance to the equator."
    answer = ["Mexico City", "Sao Paulo", "Abu Dhabi", "Austin", "Suzuka"]
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    merged_df = pd.merge(circuits_df, races_df, on="circuitId")
    merged_df = merged_df.sort_values("date", ascending=False).head(5)
    prediction = merged_df.sem_topk("What {location} is closest to the equator?", 5).location.values.tolist()
    return prediction, answer


############################################################################################################
###################################### Aggregation based queries ###########################################
############################################################################################################


def pipeline_81():
    query = "Summarize the qualities of the schools with an average score in Math under 600 in the SAT test and are exclusively virtual."
    answer = ""
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    scores_df = scores_df[scores_df["AvgScrMath"] < 600]
    schools_df = schools_df[schools_df["Virtual"] == "F"]
    merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
    prediction = merged_df.sem_agg("Summarize the qualities of the schools", all_cols=True)._output[0]
    return prediction, answer


def pipeline_82():
    query = (
        "Summarize the qualities of the schools in Riverside which the average math score for SAT is greater than 400."
    )
    answer = ""
    scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
    schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
    scores_df = scores_df[scores_df["AvgScrMath"] > 400]
    schools_df = schools_df[schools_df["City"] == "Riverside"]
    merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
    prediction = merged_df.sem_agg("Summarize the qualities of the schools", all_cols=True)._output[0]
    return prediction, answer


def pipeline_85():
    query = "Summarize common characteristics of the titles of the posts owned by the user csgillespie."
    answer = ""
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = users_df[users_df["DisplayName"] == "csgillespie"]
    merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId")
    prediction = merged_df.sem_agg("Summarize common characteristics of the {Title}s")._output[0]
    return prediction, answer


def pipeline_86():
    query = "What qualities are represented by the badges obtained by csgillespie?"
    answer = ""
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    badges_df = pd.read_csv("../pandas_dfs/codebase_community/badges.csv")
    users_df = users_df[users_df["DisplayName"] == "csgillespie"]
    merged_df = pd.merge(users_df, badges_df, left_on="Id", right_on="UserId")
    prediction = merged_df.sem_agg("What qualities are represented by the badge {Name}s")._output[0]
    return prediction, answer


def pipeline_87():
    query = "What is the average sentiment of the posts owned by the user csgillespie, Positive or Negative?"
    answer = ""
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    users_df = users_df[users_df["DisplayName"] == "csgillespie"]
    merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId")
    prediction = merged_df.sem_agg(query, all_cols=True)._output[0]
    return prediction, answer


def pipeline_88():
    query = "Summarize qualities of the comments made by user 'A Lion.'"
    answer = ""
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    users_df = users_df[users_df["DisplayName"] == "A Lion"]
    merged_df = pd.merge(users_df, comments_df, left_on="Id", right_on="UserId")
    prediction = merged_df.sem_agg("Summarize qualities of the comments", all_cols=True)._output[0]
    return prediction, answer


def pipeline_89():
    query = "Summarize the comments made on the post titled 'How does gentle boosting differ from AdaBoost?' to answer the original question"
    answer = ""
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    posts_df = posts_df[posts_df["Title"] == "How does gentle boosting differ from AdaBoost?"]
    merged_df = pd.merge(posts_df, comments_df, left_on="Id", right_on="PostId")
    prediction = merged_df.sem_agg("Summarize the comments", all_cols=True)._output[0]
    return prediction, answer


def pipeline_90():
    query = "Summarize the issues described in the comments on Neil McGuigan's posts."
    answer = ""
    users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    users_df = users_df[users_df["DisplayName"] == "Neil McGuigan"]
    merged_df = pd.merge(users_df, comments_df, left_on="Id", right_on="UserId")
    prediction = merged_df.sem_agg("Summarize the issues in the comments", all_cols=True)._output[0]
    return prediction, answer


def pipeline_91():
    query = "Summarize the comments added to the post with the highest score."
    answer = ""
    posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
    comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
    posts_df = posts_df.sort_values("Score", ascending=False).head(1)
    merged_df = pd.merge(posts_df, comments_df, left_on="Id", right_on="PostId")
    prediction = merged_df.sem_agg("Summarize the comments", all_cols=True)._output[0]
    return prediction, answer


def pipeline_92():
    query = "Summarize the track characteristics of the circuits in Germany."
    answer = ""
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    circuits_df = circuits_df[circuits_df["country"] == "Germany"]
    prediction = circuits_df.sem_agg("Summarize the track characteristics", all_cols=True)._output[0]
    return prediction, answer


def pipeline_93():
    query = "Summarize the track characteristics of the circuits in Spain."
    answer = ""
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    circuits_df = circuits_df[circuits_df["country"] == "Spain"]
    prediction = circuits_df.sem_agg("Summarize the track characteristics", all_cols=True)._output[0]
    return prediction, answer


def pipeline_94():
    query = "Provide information about the races held on Sepang International Circuit."
    answer = ""
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
    circuits_df = circuits_df[circuits_df["name"] == "Sepang International Circuit"]
    merged_df = pd.merge(circuits_df, races_df, on="circuitId")
    prediction = merged_df.sem_agg("Provide information about the races", all_cols=True)._output[0]
    return prediction, answer


def pipeline_95():
    query = "Summarize the qualities of all the drivers who finished the game in race No. 872."
    answer = ""
    drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
    results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
    results_df = results_df[results_df["raceId"] == 872]
    merged_df = pd.merge(drivers_df, results_df, on="driverId")
    prediction = merged_df.sem_agg("Summarize the qualities of the drivers", all_cols=True)._output[0]
    return prediction, answer


def pipeline_96():
    query = "Summarize the track characteristiscs of the circuits in Lisbon, Portugal."
    answer = ""
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    circuits_df = circuits_df[(circuits_df["location"] == "Lisbon") & (circuits_df["country"] == "Portugal")]
    prediction = circuits_df.sem_agg("Summarize the track characteristics", all_cols=True)._output[0]
    return prediction, answer


def pipeline_97():
    query = "Summarize the track characteristics of the US circuits."
    answer = ""
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    circuits_df = circuits_df[circuits_df["country"] == "USA"]
    prediction = circuits_df.sem_agg("Summarize the track characteristics", all_cols=True)._output[0]
    return prediction, answer


def pipeline_99():
    query = "Summarize information about the French constructors that have a lap number of over 50."
    answer = ""
    constructors_df = pd.read_csv("../pandas_dfs/formula_1/constructors.csv")
    results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
    results_df = results_df[results_df["laps"] > 50]
    constructors_df = constructors_df[constructors_df["nationality"] == "French"]
    merged_df = pd.merge(constructors_df, results_df, on="constructorId").drop_duplicates(subset="constructorId")
    prediction = merged_df.sem_agg("Summarize information about the constructors", all_cols=True)._output[0]
    return prediction, answer


def pipeline_100():
    query = "Summarize the track characteristics of the circuits in Italy."
    answer = ""
    circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
    circuits_df = circuits_df[circuits_df["country"] == "Italy"]
    prediction = circuits_df.sem_agg("Summarize the track characteristics", all_cols=True)._output[0]
    return prediction, answer


def pipeline_101():
    query = "Summarize attributes of the players with an overall rating of over 88 from 2008 to 2010?"
    answer = ""
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
    attributes_df["year"] = attributes_df["date"].str[:4].astype(int)
    attributes_df = attributes_df[
        (attributes_df["year"] >= 2008) & (attributes_df["year"] <= 2010) & (attributes_df["overall_rating"] > 88)
    ]
    merged_df = pd.merge(players_df, attributes_df, on="player_api_id").drop_duplicates(subset="player_api_id")
    prediction = merged_df.sem_agg("Summarize attributes of the players", all_cols=True)._output[0]
    return prediction, answer


def pipeline_102():
    query = "Identify common characteristics of the players whose volley score and dribbling score are over 87."
    answer = ""
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
    attributes_df = attributes_df[(attributes_df["volleys"] > 87) & (attributes_df["dribbling"] > 87)]
    merged_df = pd.merge(players_df, attributes_df, on="player_api_id").drop_duplicates(subset="player_api_id")
    prediction = merged_df.sem_agg("Identify common characteristics of the players", all_cols=True)._output[0]
    return prediction, answer


def pipeline_103():
    query = "Summarize attributes of the 10 heaviest players."
    answer = ""
    players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
    attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
    players_df = players_df.sort_values("weight", ascending=False).head(10)
    merged_df = pd.merge(players_df, attributes_df, on="player_api_id").drop_duplicates(subset="player_api_id")
    prediction = merged_df.sem_agg("Summarize attributes of the players", all_cols=True)._output[0]
    return prediction, answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", nargs="+", required=True, help="List of pipelines to run")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    lm = OpenAIModel(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        api_base="http://localhost:8000/v1",
        provider="vllm",
        max_tokens=1024,
        max_batch_size=512,
    )
    lotus.settings.configure(lm=lm)

    args = parse_args()
    if args.verbose:
        lotus.logger.setLevel("DEBUG")

    pipelines = args.pipelines
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for pipeline in pipelines:
        print(f"Running pipeline {pipeline}")
        tic = time.time()
        prediction, answer = eval(f"pipeline_{pipeline}")()
        latency = time.time() - tic
        output = {"prediction": prediction, "answer": answer, "latency": latency, "query_id": pipeline}
        print(output)

        if args.output_dir:
            with open(os.path.join(args.output_dir, f"query_{pipeline}.json"), "w+") as fp:
                json.dump(output, fp)
