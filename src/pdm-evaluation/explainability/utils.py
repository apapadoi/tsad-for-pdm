import pandas as pd


def episodes_formulation(data ,datetime_column ,event_indicator=None ,maintenance_list=None ,failure_list=None
                         ,event_df=None ,source_column='source' ,DIVIDER=3600):

    if event_df is not None:
        # source
        if datetime_column not in event_df.columns or source_column not in event_df.columns or "code" not in event_df.columns:
            raise ValueError("datetime_column and source_column must be present in event data.")
        maintenance_list =set(maintenance_list).difference(failure_list)
        maintenance_col ="maintenance_event"
        failure_col ="failure_event"
        event_df[maintenance_col ] =[1 if code in maintenance_list else 0 for code in event_df["code"].values]
        event_df[failure_col ] =[1 if code in failure_list else 0 for code in event_df["code"].values]
        event_df[datetime_column ] =pd.to_datetime(event_df[datetime_column])
        event_data = event_df[[datetime_column, source_column, maintenance_col, failure_col]].copy()
        if datetime_column not in event_df.columns or source_column not in event_df.columns:
            raise ValueError("datetime_column and source_column must be present in data.")

        all_sources = []
        all_episodes = []
        all_run_to_failure = []
        original_s_has_f = {}
        for source in data[source_column].unique():
            df_source = data[data[source_column] == source].copy()

            episodes, rtfs, new_sources = data_split_by_event(df_source,
                                                              event_data[event_data[source_column] == source].copy(),
                                                              datetime_column, failure_col, maintenance_col,
                                                              source_column)
            all_episodes.extend(episodes)
            all_run_to_failure.extend(rtfs)
            original_s_has_f[source] = max(rtfs) == 1 or original_s_has_f.get(source, False)
            all_sources.extend(new_sources)

        return all_episodes, all_run_to_failure, all_sources, original_s_has_f


    elif event_indicator is not None:
        # group by source and indicate the event:
        all_episodes = []
        all_run_to_failure = []
        all_sources = []
        original_s_has_f ={}
        for source ,group_df in data.groupby(source_column):
            group_df =group_df.sort_values(by=datetime_column).reset_index(drop=True)
            if len(group_df[event_indicator].unique() ) >1:
                raise ValueError \
                    (f"event_indicator column must be binary (0 and 1) for each source. Source {source} has values {group_df[event_indicator].unique()}")
            if "RUL" not in group_df.columns:
                maxdate = group_df[datetime_column].max()
                group_df["RUL"] = [(maxdate - dtime).total_seconds() / DIVIDER for dtime in
                                   group_df[datetime_column]]

            all_run_to_failure.append(group_df.iloc[0][event_indicator])
            original_s_has_f[source ] =group_df.iloc[0][event_indicator ]==1  or original_s_has_f.get(source ,False)
            all_episodes.append(group_df.drop(columns=[event_indicator]))
            all_sources.append(f"{source}_ep0")
        return all_episodes, all_run_to_failure ,all_sources ,original_s_has_f
    # check if maintenance_column and failure_column are in data
    else:
        print \
            ("Warning: event column is not in data and not eventDf was given, we consider each source as run_to_failure.")
        all_episodes = []
        all_run_to_failure = []
        original_s_has_f = {}
        all_sources = []
        for source, group_df in data.groupby(source_column):
            group_df = group_df.sort_values(by=datetime_column).reset_index(drop=True)
            if "RUL" not in group_df.columns:
                maxdate =group_df[datetime_column].max()
                group_df["RUL" ] =[(maxdate - dtime).total_seconds( ) /DIVIDER for dtime in group_df[datetime_column]]
            all_run_to_failure.append(1)
            original_s_has_f[source] = True or original_s_has_f.get(source, False)
            all_episodes.append(group_df)
            all_sources.append(f"{source}_ep0")
        return all_episodes, all_run_to_failure, all_sources, original_s_has_f


    # # check if datetime_column, source_column are in data
    # if datetime_column not in data.columns or source_column not in data.columns:
    #     raise ValueError("datetime_column and source_column must be present in data.")
    #
    # all_sources=[]
    # all_episodes = []
    # all_run_to_failure = []
    # original_s_has_f = {}
    # for source in data[source_column].unique():
    #     df_source=data[data[source_column]==source].copy()
    #
    #     episodes, rtfs,new_sources = data_split_by_event(df_source,event_data[event_data[source_column]==source],
    #                                                      datetime_column,failure_column,maintenance_column,source_column,DIVIDER)
    #     all_episodes.extend(episodes)
    #     all_run_to_failure.extend(rtfs)
    #     original_s_has_f[source] = max(rtfs)==1 or original_s_has_f.get(source, False)
    #     all_sources.extend(new_sources)
    #
    # return all_episodes, all_run_to_failure,all_sources,original_s_has_f




def data_split_by_event(df_source ,event_source ,datetime_column ,failure_column ,maintenance_column
                        ,source_column='source' ,DIVIDER=3600):
    df_source.sort_values(by=datetime_column, inplace=True)
    df_source.reset_index(drop=True, inplace=True)
    event_source.sort_values(by=datetime_column, inplace=True)
    event_source.reset_index(drop=True, inplace=True)
    episodes = []
    rtfs = []
    new_sources =[]
    counter =0
    for idx, event_row in event_source.iterrows():
        event_time = event_row[datetime_column]
        found = None
        if event_row[failure_column] == 1:
            # Failure event
            found =1
        elif event_row[maintenance_column] == 1:
            found = 0
        if found is not None:
            if idx == 0:
                start_time = df_source[datetime_column].min()
            else:
                prev_event_time = event_source.loc[idx - 1, datetime_column]
                start_time = prev_event_time
            end_time = event_time
            episode = df_source[(df_source[datetime_column] > start_time) & (df_source[datetime_column] <= end_time)].copy()
            episode["RUL" ]= [(episode[datetime_column].max() - dtime).total_seconds( ) /DIVIDER for dtime in episode[datetime_column]]
            episode[source_column ] =f"{df_source.iloc[0][source_column]}_ep{counter}"

            if episode.shape[0] == 0:
                continue

            episodes.append(episode)
            rtfs.append(found)
            new_sources.append(f"{df_source.iloc[0][source_column]}_ep{counter}")
            counter += 1
    return episodes, rtfs, new_sources


def prepare():
    df=pd.read_csv("EDP/EDP_combined.csv")
    df["Timestamp"]=pd.to_datetime(df["Timestamp"])
    event_df=pd.read_csv("EDP/failures.csv")
    event_df["Timestamp"]=pd.to_datetime(event_df["Timestamp"])
    episodes, run_to_failure, _, original_s_has_f = episodes_formulation(df, "Timestamp", None,
                                                                         [],
                                                                         ["failure"], event_df, "source",
                                                                         3600)


    return episodes, run_to_failure
