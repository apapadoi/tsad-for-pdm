import pandas as pd



globalpath="."
def failureandEvents(events,excluded,tdcevents,sources,exresets=[]):

    failuretimes = [ev['dt'] for i, ev in events.iterrows() if
                    ev["action_type"] == "R" and str(ev["vehicle_id"]) in sources and notinlist(excluded, ev["desc"])]
    failurecodes = [ev['desc'] for i, ev in events.iterrows() if
                    ev["action_type"] == "R" and str(ev["vehicle_id"]) in sources and notinlist(excluded, ev["desc"])]
    failuresources = [ev['vehicle_id'] for i, ev in events.iterrows() if
                      ev["action_type"] == "R" and str(ev["vehicle_id"]) in sources and notinlist(excluded, ev["desc"])]

    eventsofint = [ev['dt'] for i, ev in events.iterrows() if ((ev["action_type"] == "S" and notinlist(exresets, ev["desc"])) or (
            ev["action_type"] == "R" and notinlist(excluded, ev["desc"]) == False and notinlist(exresets, ev["desc"]))) and str(
        ev["vehicle_id"]) in sources]
    eventsofint.extend([ev['dt'] for i, ev in tdcevents.iterrows() if str(ev["vehicle_id"]) in sources])

    eventsofintsources = [ev['vehicle_id'] for i, ev in events.iterrows() if((ev["action_type"] == "S" and notinlist(exresets, ev["desc"])) or (
            ev["action_type"] == "R" and notinlist(excluded, ev["desc"]) == False and notinlist(exresets, ev["desc"])))  and str(ev["vehicle_id"]) in sources]
    eventsofintsources.extend([ev['vehicle_id'] for i, ev in tdcevents.iterrows() if str(ev["vehicle_id"]) in sources])

    eventscodes = [ev['desc'] for i, ev in events.iterrows() if ((ev["action_type"] == "S" and notinlist(exresets, ev["desc"])) or (
            ev["action_type"] == "R" and notinlist(excluded, ev["desc"]) == False and notinlist(exresets, ev["desc"]))) and str(
        ev["vehicle_id"]) in sources]
    eventscodes.extend([ev['dtc_status'] for i, ev in tdcevents.iterrows() if str(ev["vehicle_id"]) in sources])
    return failuretimes,failurecodes,failuresources,eventsofint,eventsofintsources,eventscodes

def formulateData(dfs, events, tdcevents, sources=[], excluded=["Accident", "Tyre"], reseteventsParameter=["Standard service", "Oil change"]):


    unicodes = [f"{ev['action_type']}_{ev['desc']}" for i, ev in events.iterrows() if
                (ev["action_type"] == "R" and notinlist(excluded, ev["desc"])) or ev["desc"] in reseteventsParameter]
    unicodes = list(set(unicodes))
    alltimes = []
    alltypesofdata = []
    resetscodeslis = []
    for dff, sourccc in zip(dfs, sources):
        alltimes.extend([i for i in dff.index])
        alltypesofdata.extend([f"data:{sourccc}" for i in dff.index])
        resetscodeslis.append([(code, sourccc) for code in unicodes])
        dff.index = [dt.tz_localize(None) for dt in dff.index]

    events['dt'] = [dt.tz_localize(None) for dt in events['dt']]
    alltimes.extend([i for i in events["dt"]])
    alltypesofdata.extend(["maintenance" for i in events.index])

    tdcevents['dt'] = [dt.tz_localize(None) for dt in tdcevents['dt']]
    alltimes.extend([i for i in tdcevents["dt"]])
    alltypesofdata.extend(["dtc" for i in tdcevents.index])


    alltimes = [dt.tz_localize(None) for dt in alltimes]

    time_type = list(set(zip(alltimes, alltypesofdata)))

    timee = [x for x, _ in sorted(time_type)]
    typee = [y for _, y in sorted(time_type)]

    return timee, typee, resetscodeslis, events, tdcevents, dfs
def check_if_events_exist(datadf,vehicle,unicodes):
    df = pd.read_csv(f"{globalpath}/DataFolder/Navarchos/newerservices.csv")
    df['dt'] = pd.to_datetime(df['dt'], format='mixed')

    vehicle_related = df[df['vehicle_id'] == int(vehicle)]
    begin = min(datadf.index)
    end = max(datadf.index)
    tupss = [(des, time) for des, time in zip(vehicle_related['desc'], vehicle_related['dt']) if
             time >= begin and (f"R_{des}" in unicodes or f"S_{des}" in unicodes)]
    tupss = list(set(tupss))
    return len(tupss)>0


def notinlist(listoroi, desc):
    for oros in listoroi:
        if oros in desc:
            return False
    return True

def localNavarchosSimulation(datasetname = "n=300_slide=50",oilInReset=False,ExcludeNoInformationVehicles=False,noserviceResset=False):
    if ExcludeNoInformationVehicles:
        sources = ['25', '5', '4', '16', '28', '27', '14', '20', '26', '33', '24', '18', '29', '31', '2', '30', '7',
                   '21', '13', '17', '34', '32', '23', '11', '8', '9']
    else:
        sources = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                   '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                   '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

    excluded = ["Accident", "Tyre", "door", "Whell", "Wheel", "Rims", "Lamp", "Horn", "Strut", "Αντιστάσεις",
                "Windsfield",
                "DPF", "Blue", "blue", "brake", "Battery", "Brake", "Επιδι", "Steer", "Shock",
                "motor driven fan", "Spark", "Engine base", "Mirror"]

    excludedresets = ["Accident", "Tyre", "door", "Whell", "Wheel", "Rims", "Lamp", "Horn", "Strut", "Αντιστάσεις",
                      "Windsfield",
                      "DPF", "Blue", "blue", "brake", "Battery", "Brake", "Επιδι", "Steer", "Shock", "Spark"]

    ## NOW: RESET ON OIL CHANGE AND ACCIDENTS
    if oilInReset:
        reseteventsParameter = ["Standard service", "Oil change"]
    else:
        reseteventsParameter = ["Standard service"]

    events = pd.read_csv(f"{globalpath}/DataFolder/Navarchos/newerservices.csv", index_col=0)
    events['dt'] = pd.to_datetime(events['dt'])  # , format='mixed')
    unicodes = [f"{ev['action_type']}_{ev['desc']}" for i, ev in events.iterrows() if
                (ev["action_type"] == "R" and notinlist(excluded, ev["desc"])) or ev["desc"] in reseteventsParameter]
    unicodes = list(set(unicodes))
    dfs = []

    sourceori = []
    for vehicle in sources:
        datadf = pd.read_csv(f"{globalpath}/DataFolder/Navarchos/{datasetname}/{vehicle}.csv", index_col=0)

        datadf.index = pd.to_datetime(datadf.index)
        datadf = datadf[~datadf.index.duplicated(keep='first')]

        sourceori.append(vehicle)
        dfs.append(datadf)

    sources = sourceori
    print(f"Total vehcicles: {len(sources)}")
    print(sources)
    events = pd.read_csv(f"{globalpath}/DataFolder/Navarchos/newerservices.csv", index_col=0)
    events['dt'] = pd.to_datetime(events['dt'])  # , format='mixed')
    # print(events.head())

    tdcevents = pd.read_csv(f"{globalpath}/DataFolder/Navarchos/dtc_all.csv", index_col=0)
    tdcevents['dt'] = pd.to_datetime(tdcevents['dt'])

    timee, typee, resetscodeslis, events, tdcevents, dfs = formulateData(dfs, events, tdcevents,
                                                                         sources=sources,
                                                                         excluded=excludedresets,
                                                                         reseteventsParameter=reseteventsParameter)
    failuretimes, failurecodes, failuresources, eventsofint, eventsofintsources, eventscodes = failureandEvents(events,
                                                                                                                excluded,
                                                                                                                tdcevents,
                                                                                                                sources)

    correctfailuretimes=[]
    correctfailuresources=[]
    correctfailurecodes=[]
    for ft,fs,fc in zip(failuretimes,failuresources,failurecodes):
        if ft > dfs[sourceori.index(str(fs))].index[0]:
            correctfailuretimes.append(ft)
            correctfailuresources.append(fs)
            correctfailurecodes.append(fc)
    failuretimes=correctfailuretimes
    failuresources=correctfailuresources
    failurecodes=correctfailurecodes

    eventdata={
        "date":[],
        "type":[],
        "source":[],
        "description":[],
    }
    for ind,evrow in events.iterrows():
        for ev in resetscodeslis:
            if ev[0] == evrow["desc"] and ev[1] == evrow["source"]:
                eventdata["date"].append(evrow["dt"])
                eventdata["type"].append("reset")
                eventdata["source"].append(str(evrow["source"]))
                eventdata["description"].append(evrow["desc"])
                break
    for ftime,fcode,fsource in zip(failuretimes,failurecodes,failuresources):
        eventdata["date"].append(ftime)
        eventdata["type"].append("fail")
        eventdata["source"].append(str(fsource))
        eventdata["description"].append(fcode)

    finaldfs=[]
    for datadf in dfs:
        datadf["dt"] = [kati for kati in datadf.index]
        datadf = datadf.reset_index(drop=True)
        finaldfs.append(datadf)


    dfevents=pd.DataFrame(eventdata)
    dfevents=dfevents.sort_values(by=['date'])
    return sourceori,finaldfs,dfevents













    # historic_data = []
    #
    # sourceori,dfs,dfevents=localNavarchosSimulation(datasetname="n=300_slide=100", all_sources=True, oilInReset=False,
    #                              ExcludeNoInformationVehicles=True, noserviceResset=False)
    #
    # target_data=dfs
    #
    # event_data = dfevents
    #
    # event_preferences: EventPreferences = {
    #     'failure': [
    #         EventPreferencesTuple(description='*', type='fail', source='*', target_sources='=')
    #     ],
    #     'reset': [
    #         EventPreferencesTuple(description='*', type='reset', source='*', target_sources='=')
    #     ]
    # }
    #
    # my_pipeline = PdMPipeline(
    #     steps={
    #         'preprocessor': DefaultPreProcessor,
    #         'method': ProfileBased,
    #         'postprocessor': MovingAveragePostProcessor,
    #         'thresholder': ConstantThresholder,
    #     },
    #     event_data=event_data,
    #     event_preferences=event_preferences,
    #     dates='dt',
    #     predictive_horizon='30 days',
    #     lead='1 days',
    #     beta=1
    # )
    # name="Navarchos_good"
    # my_experiment = AutoProfileSemiSupervisedPdMExperiment(
    #     experiment_name='Navarchos_good',
    #     target_data=target_data,
    #     target_sources=sourceori,
    #     pipeline=my_pipeline,
    #     profile_size=15,
    #     param_space=dict(
    #         thresholder_threshold_value=[0.5, 0.25, 0.75],
    #         postprocessor_window_length=[15, 5, 10]
    #     ),
    #     num_iteration=1,
    #     n_jobs=7,
    #     initial_random=2,
    #     artifacts='my_artifacts'
    # )


    #my_experiment.execute()

