// Power Query (M) template for the ED Arrival Outlook Dataflow Gen2 query.
//
// Create a text parameter named DailyArrivalOutlookUrl containing the direct/raw
// Dropbox URL for /daily_arrival_outlook.csv, then paste this query into a blank
// query in Dataflow Gen2. The pipeline itself remains the source of truth for the
// Montreal-local horizon_day field, so Power BI does not need to infer "today"
// from UTC service time.
let
    Source = Csv.Document(
        Web.Contents(DailyArrivalOutlookUrl),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        PromotedHeaders,
        {
            {"target_date", type date},
            {"generated_at_local", type datetimezone},
            {"forecast_stage", type text},
            {"horizon_day", Int64.Type},
            {"predicted_arrivals", type number},
            {"lower_80", type number},
            {"upper_80", type number},
            {"observed_arrivals", type number},
            {"expected_remaining", type number},
            {"seasonal_weekday_baseline", type number},
            {"delta_vs_baseline", type number},
            {"top_driver_1", type text},
            {"top_driver_1_effect", type number},
            {"top_driver_2", type text},
            {"top_driver_2_effect", type number},
            {"top_driver_3", type text},
            {"top_driver_3_effect", type number},
            {"explainability_method", type text},
            {"explanation_text", type text},
            {"source_model", type text},
            {"data_cutoff", type text},
            {"model_version", type text}
        },
        "en-CA"
    ),
    ValidRows = Table.SelectRows(
        Typed,
        each [target_date] <> null
            and [predicted_arrivals] <> null
            and [lower_80] <> null
            and [upper_80] <> null
    ),
    IntervalInvariant = Table.AddColumn(
        ValidRows,
        "interval_valid",
        each [lower_80] <= [predicted_arrivals] and [predicted_arrivals] <= [upper_80],
        type logical
    ),
    Checked = if List.Contains(IntervalInvariant[interval_valid], false)
        then error "daily_arrival_outlook.csv contains an invalid prediction interval"
        else IntervalInvariant,
    Sorted = Table.Sort(Checked, {{"target_date", Order.Ascending}})
in
    Sorted
