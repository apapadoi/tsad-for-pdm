# Copyright 2026 Anastasios Papadopoulos, Apostolos Giannoulidis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import Normalize

def show_TOP_feature(TOP_K = 45,particular_sources=None):
    with open("Pickles/EDP/IF_c_kshap_piv_50_impo.pkl", "rb") as f:
        data = pickle.load(f)


    names=[f"f_{i}" for i in range(len(data["important_features"][0]))]
    # names=[
    #     "Gen_RPM_Max", "Gen_RPM_Min", "Gen_RPM_Avg", "Gen_RPM_Std", "Gen_Bear_Temp_Avg",
    #     "Gen_Phase1_Temp_Avg", "Gen_Phase2_Temp_Avg", "Gen_Phase3_Temp_Avg", "Hyd_Oil_Temp_Avg",
    #     "Gear_Oil_Temp_Avg", "Gear_Bear_Temp_Avg", "Nac_Temp_Avg", "Rtr_RPM_Max", "Rtr_RPM_Min",
    #     "Rtr_RPM_Avg", "Amb_WindSpeed_Max", "Amb_WindSpeed_Min", "Amb_WindSpeed_Avg",
    #     "Amb_WindSpeed_Std", "Amb_WindDir_Relative_Avg", "Amb_WindDir_Abs_Avg", "Amb_Temp_Avg",
    #     "Prod_LatestAvg_ActPwrGen0", "Prod_LatestAvg_ActPwrGen1", "Prod_LatestAvg_ActPwrGen2",
    #     "Prod_LatestAvg_TotActPwr", "Prod_LatestAvg_ReactPwrGen0", "Prod_LatestAvg_ReactPwrGen1",
    #     "Prod_LatestAvg_ReactPwrGen2", "Prod_LatestAvg_TotReactPwr", "HVTrafo_Phase1_Temp_Avg",
    #     "HVTrafo_Phase2_Temp_Avg", "HVTrafo_Phase3_Temp_Avg", "Grd_InverterPhase1_Temp_Avg",
    #     "Cont_Top_Temp_Avg", "Cont_Hub_Temp_Avg", "Cont_VCP_Temp_Avg", "Gen_SlipRing_Temp_Avg",
    #     "Spin_Temp_Avg", "Blds_PitchAngle_Min", "Blds_PitchAngle_Max", "Blds_PitchAngle_Avg",
    #     "Blds_PitchAngle_Std", "Cont_VCP_ChokcoilTemp_Avg", "Grd_RtrInvPhase1_Temp_Avg",
    #     "Grd_RtrInvPhase2_Temp_Avg", "Grd_RtrInvPhase3_Temp_Avg", "Cont_VCP_WtrTemp_Avg",
    #     "Grd_Prod_Pwr_Avg", "Grd_Prod_CosPhi_Avg", "Grd_Prod_Freq_Avg", "Grd_Prod_VoltPhse1_Avg",
    #     "Grd_Prod_VoltPhse2_Avg", "Grd_Prod_VoltPhse3_Avg", "Grd_Prod_CurPhse1_Avg",
    #     "Grd_Prod_CurPhse2_Avg", "Grd_Prod_CurPhse3_Avg", "Grd_Prod_Pwr_Max", "Grd_Prod_Pwr_Min",
    #     "Grd_Busbar_Temp_Avg", "Rtr_RPM_Std", "Amb_WindSpeed_Est_Avg", "Grd_Prod_Pwr_Std",
    #     "Grd_Prod_ReactPwr_Avg", "Grd_Prod_ReactPwr_Max", "Grd_Prod_ReactPwr_Min",
    #     "Grd_Prod_ReactPwr_Std", "Grd_Prod_PsblePwr_Avg", "Grd_Prod_PsblePwr_Max",
    #     "Grd_Prod_PsblePwr_Min", "Grd_Prod_PsblePwr_Std", "Grd_Prod_PsbleInd_Avg",
    #     "Grd_Prod_PsbleInd_Max", "Grd_Prod_PsbleInd_Min", "Grd_Prod_PsbleInd_Std",
    #     "Grd_Prod_PsbleCap_Avg", "Grd_Prod_PsbleCap_Max", "Grd_Prod_PsbleCap_Min",
    #     "Grd_Prod_PsbleCap_Std", "Gen_Bear2_Temp_Avg", "Nac_Direction_Avg"
    # ]
    labels = data["label"]
    timestamps = [pd.to_datetime(date) for date in data["timestamp"]]
    sources = data["source"]
    score = data["score"]
    important_features = [[(nam,imp) for imp,nam in  zip(kati,names) if abs(imp)>0.05] for kati in data["important_features"]]



    # -----------------------------
    # Step 1: Build DataFrame
    # -----------------------------
    dfall = pd.DataFrame({
        "timestamp": timestamps,
        "label": labels,
        "source": sources,
        "important_features": important_features,
        "score": score
    })


    # Ensure timestamps are datetime and sorted
    dfall["timestamp"] = pd.to_datetime(dfall["timestamp"])
    dfall = dfall.sort_values("timestamp")


    if particular_sources is not None:
        sources=particular_sources
    else:
        # sources=set([source for source in dfall["source"].unique()])
        sources=set([source.split("_")[0] for source in dfall["source"].unique()])

    print(f"All sources {dfall['source'].unique()}")
    for source in sources:
            if particular_sources is not None:
                df = dfall[dfall["source"].isin(source)].copy()
            else:
                df=dfall[dfall["source"].str.startswith(source)].copy()
            failures=[]
            print(df["source"].unique())
            for episodeid in df["source"].unique():
                ep_df=df[df["source"]==episodeid]
                failures.append(ep_df["timestamp"].max())

            # -----------------------------
            # Step 2: Compute global top features
            # -----------------------------
            all_features = [feat[0] for sublist in df["important_features"] for feat in sublist]
            feature_counts = Counter(all_features)

            top_features = [f for f, _ in feature_counts.most_common(TOP_K)]
            maximp = max([abs(f[1].item()) for feat in df["important_features"].tolist() for f in feat])
            # -----------------------------
            # Step 3: Aggregate by time window (e.g., per day)
            # -----------------------------
            df["date"] = df["timestamp"]

            counts_over_time = {f: [] for f in top_features}
            dictfeats = {}

            for d , feats in zip(df["timestamp"],df["important_features"]):
                for f in feats:
                    # dictfeats[f[0]]=f[1].item()/maximp
                    dictfeats[f[0]]=f[1].item()
                for f in top_features:
                    counts_over_time[f].append(dictfeats.get(f, 0))

            # -----------------------------
            # Step 4: Create heatmap DataFrame
            # -----------------------------
            df_heat = pd.DataFrame(counts_over_time, index=df["timestamp"].tolist())


            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler(feature_range=(-1, 1))
            # df_heat[" "]=[np.nan for _ in range(df_heat.shape[0])]

            score_to_plot= scaler.fit_transform(np.array(df["score"]).reshape(-1, 1)).flatten()
            # df_heat["PH"] = [lab * 2 - 1 for lab in df["label"]]
            # -----------------------------
            # Step 5: Plot heatmap
            # -----------------------------
            fig = plt.figure(figsize=(10, 7))
            gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[3, 1], width_ratios=[2, 1])

            # Left column: two plots stacked (3:1 ratio)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

            # Right column: one plot spanning both rows
            ax3 = fig.add_subplot(gs[:, 1])
            ax2.sharex(ax1)

            # sns.heatmap(df_heat.T, cmap="coolwarm",center=0, yticklabels=True,ax=ax1)
            # ##ax1.set_yticks(ticks=range(len(df_heat.T.index)), labels=df_heat.T.index, fontsize=8)
            # ## ax1.tick_params(labelbottom=False)
            # # ax1.set_title(f"Heatmap of Important Features Over Time {source}")
            # # ax1.set_xlabel("Time")
            # # ax1.set_ylabel("Feature")
            # ax1.set_xticklabels([])
            times = df['timestamp'].tolist()

            norm = Normalize(vmin=df_heat.values.min(), vmax=df_heat.values.max(), clip=True)

            im = ax1.imshow(
                df_heat.T.values,
                aspect='auto',
                cmap='coolwarm',
                # vmin=-1,
                # vmax=1,
                norm=norm,
                interpolation='none',
                extent=[mdates.date2num(times[0]), mdates.date2num(times[-1]), 0, df_heat.shape[1]]
            )

            # ax1.set_yticks(range(len(df_heat.columns)))
            ax1.set_yticks(np.arange(df_heat.shape[1]) + 0.5)
            ax1.set_yticklabels(df_heat.columns, fontsize=10)
            ax1.set_ylabel("Feature")
            ax1.tick_params(labelbottom=False)

            # Add colorbar
            colorbar = fig.colorbar(im, ax=ax1, orientation='horizontal', location='top',
                                    shrink=0.7)
            # colorbar.set_ticks([-1,0, 1])  # Set ticks at the edges
            # colorbar.set_ticklabels(['Negative','Zero','Positive'])


            # score_to_plot = pd.Series(score_to_plot).rolling(window=5, center=True).mean().fillna(
            #     method='bfill').fillna(method='ffill')
            ax2.plot(df['timestamp'], score_to_plot, label='Scores', color="black")
            for i,fail in enumerate(failures):
                if i==0:
                    ax2.axvline(x=fail, color='red', linewidth=4, label='Failure')
                else:
                    ax2.axvline(x=fail, color='red', linewidth=4)
            # ax2.set_xlabel("Time")
            ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=8)
            ax2.set_xlabel(f"b) Anomaly scores")
            sourceToplot=source
            if isinstance(sourceToplot,list):
                sourceToplot=sourceToplot[0].split("_")[0]
            ax1.set_xlabel(f"a) Heatmap of important features over time for source {sourceToplot}")
            ax2.legend(ncols=2)
            global_importance = df_heat.mean(axis=0)

            # Plot global importance on ax3
            ax3.barh(global_importance.index, global_importance.values, color='blue')


            # Normalize the global importance values
            norm = Normalize(vmin=global_importance.min(), vmax=global_importance.max())
            colors = cm.coolwarm(norm(global_importance.values))

            # Use the colors for the bars
            ax3.barh(global_importance.index, global_importance.values, color=colors)
            # ax3.set_title("")
            ax3.set_xlabel("c) Global importance of features")
            ax3.set_ylabel("Feature")
            ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=10)
            ax3.invert_yaxis()  # Invert y-axis for better readability



            plt.show()


if __name__=="__main__":
    show_TOP_feature(TOP_K=10,particular_sources=[["T07_ep0","T07_ep1","T07_ep2"]])