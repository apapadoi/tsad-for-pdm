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

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table


# data = [list1, list2, list3]
#
# # Create the boxplot
# plt.boxplot(data)
#
# # Add titles and labels
# plt.title('Box Plot for Three Lists')
# plt.xlabel('List')
# plt.ylabel('Values')
# plt.xticks([1, 2, 3], ['List 1', 'List 2', 'List 3'])

def allDatasetsFlavorsAvg(df):
    dftemp = df[df["Method"] == "Avg."]


    dftemp = dftemp[["Online", "Sliding","Unsupervised"]]
    print(dftemp.head(10))

    result = autorank(dftemp, alpha=0.05, verbose=False, force_mode="nonparametric")
    create_report(result)
    plot_stats(result, allow_insignificant=True)
    plt.show()

def Methods_all_datasets_all_flavors(df):
    allscores=[]
    allnames=[]
    for techname in df["Method"].unique():
        if "Avg" in techname:
            continue
        dftemp = df[df["Method"] == techname]
        scores=[]
        for col in ["Online", "Sliding","Historical","Unsupervised"]:
            scores.extend([perf for perf in dftemp[col].values if np.isnan(perf)==False])
        allscores.append(scores)
        allnames.append(techname)
        print(f"{techname} : {scores}")
    medians = [np.median(lst) for lst in allscores]

    # Sort data, labels, and colors by median values
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    # Add titles and labels
    fig, ax = plt.subplots()
    box = ax.boxplot(allscores, patch_artist=True)

    # Define colors
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', '#800080',
              '#008080', '#000080']
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]
    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    plt.title('Box Plot (considering all flavors)')
    plt.xlabel('Methods')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames)+1), allnames)

    plt.show()



def Methods_all_datasets_best_flavor(df):
    allscores = []
    allnames = []
    for techname in df["Method"].unique():
        if "Avg" in techname:
            continue
        dftemp = df[df["Method"] == techname]
        scores = []
        for dataset in dftemp["Dataset"].unique():
            deftemp_dataset=dftemp[dftemp["Dataset"] == dataset]
            collect_scores=[]
            for col in ["Online", "Sliding", "Historical", "Unsupervised"]:
                collect_scores.extend([perf for perf in deftemp_dataset[col].values if np.isnan(perf) == False])
            if len(collect_scores)>=1:
                scores.append(max(collect_scores))
        allscores.append(scores)
        allnames.append(techname)
        print(f"{techname} : {scores}")

    medians = [np.median(lst) for lst in allscores]
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', '#800080',
              '#008080', '#000080']
    # Sort data, labels, and colors by median values
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]

    # Add titles and labels
    fig, ax = plt.subplots()
    box = ax.boxplot(allscores, patch_artist=True)

    # Define colors


    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    plt.title('Box Plot (choosing the best flavor each time)')
    plt.xlabel('Methods')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames)

    plt.show()

def flavors_all_datasets_on_common_tecn_semi_Online_Sliding_historic_and_Usupervised(df):
    allscores = []
    allnames = []

    df = df.dropna(subset=["Historical"])

    for flavor in ["Online", "Sliding", "Historical","Unsupervised"]:
        dftemp=df[df["Method"].isin(["lof","np","knn","if"])]
        scores= [perf for perf in dftemp[flavor].values if np.isnan(perf) == False]
        allscores.append(scores)
        allnames.append(flavor)
        print(f"{flavor} : {scores}")
    medians = [np.median(lst) for lst in allscores]

    # Sort data, labels, and colors by median values
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    # Add titles and labels
    fig, ax = plt.subplots()
    box = ax.boxplot(allscores, patch_artist=True)

    # Define colors
    colors = ['#FF0000', '#00FF00', '#0000FF']
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]
    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    plt.title('Box Plot flavors across all datasets')
    plt.xlabel('Methods')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames)

    plt.show()

    critical_diagram(allnames, allscores)


def flavors_all_datasets_on_common_tecn_semi_Online_Sliding_historic(df):
    allscores = []
    allnames = []

    df = df.dropna(subset=["Historical"])

    for flavor in ["Online", "Sliding", "Historical"]:
        dftemp=df[df["Method"] != "sand"]
        scores= [perf for perf in dftemp[flavor].values if np.isnan(perf) == False]
        allscores.append(scores)
        allnames.append(flavor)
        print(f"{flavor} : {scores}")
    medians = [np.median(lst) for lst in allscores]

    # Sort data, labels, and colors by median values
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    # Add titles and labels
    fig, ax = plt.subplots()
    box = ax.boxplot(allscores, patch_artist=True)

    # Define colors
    colors = ['#FF0000', '#00FF00', '#0000FF']
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]
    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    plt.title('Box Plot flavors across all datasets')
    plt.xlabel('Methods')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames)

    plt.show()

    critical_diagram(allnames, allscores)
def flavors_all_datasets_on_common_tecn_semi_Online_Sliding(df):
    allscores = []
    allnames = []
    for flavor in ["Online", "Sliding"]:
        dftemp=df[df["Method"] != "sand"]
        scores= [perf for perf in dftemp[flavor].values if np.isnan(perf) == False]

        allscores.append(scores)
        allnames.append(flavor)
        print(f"{flavor} : {scores}")

    colors = ['#FF0000', '#00FF00']
    boxplots(allnames,allscores,colors)
    critical_diagram(allnames, allscores)
def flavors_all_datasets_on_common_tecn(df):
    allscores = []
    allnames = []

    for flavor in ["Online", "Sliding", "Unsupervised"]:
        dftemp=df[df["Method"].isin(["knn","lof","np","if"])]
        scores= [perf for perf in dftemp[flavor].values if np.isnan(perf) == False]

        allscores.append(scores)
        allnames.append(flavor)
        print(f"{flavor} : {scores}")

    colors = ['#FF0000', '#00FF00', '#FFFF00']
    boxplots(allnames,allscores,colors)


    critical_diagram(allnames, allscores)

def flavors_all_datasets_on_common_tecn_and_historical(df):
    allscores = []
    allnames = []

    df = df.dropna(subset=["Historical"])

    for flavor in ["Online", "Sliding", "Historical", "Unsupervised"]:
        dftemp=df[df["Method"].isin(["knn","lof","np","if"])]
        scores= [perf for perf in dftemp[flavor].values if np.isnan(perf) == False]
        allscores.append(scores)
        allnames.append(flavor)
        print(f"{flavor} : {scores}")
    medians = [np.median(lst) for lst in allscores]

    # Sort data, labels, and colors by median values
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    # Add titles and labels
    fig, ax = plt.subplots()
    box = ax.boxplot(allscores, patch_artist=True)

    # Define colors
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]
    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    plt.title('Box Plot flavors across all datasets')
    plt.xlabel('Methods')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames)

    plt.show()

    critical_diagram(allnames, allscores)


def flavors_all_datasets_dataset_wit_historical_only_Semi(df):
    allscores = []
    allnames = []

    df = df.dropna(subset=["Historical"])

    for flavor in ["Online", "Sliding", "Historical"]:
        dftemp = df[df["Method"]!="Avg."]
        scores = [perf for perf in dftemp[flavor].values if np.isnan(perf) == False]
        allscores.append(scores)
        allnames.append(flavor)
        print(f"{flavor} : {scores}")
    medians = [np.median(lst) for lst in allscores]

    # Sort data, labels, and colors by median values
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    # Add titles and labels
    fig, ax = plt.subplots()
    box = ax.boxplot(allscores, patch_artist=True)

    # Define colors
    colors = ['#FF0000', '#00FF00', '#0000FF']
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]
    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    plt.title('Box Plot flavors across all datasets')
    plt.xlabel('Methods')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames)

    plt.show()

    critical_diagram(allnames, allscores)


def flavors_per_dataset(df):

    allcolors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
    fig, ax = plt.subplots(len(df["Dataset"].unique()),1)
    counter=-1
    plt.title("Flavor per dataset")
    for dataset in df["Dataset"].unique():
        allscores = []
        allnames = []
        colors = []
        counter +=1
        dfdataset=df[df["Dataset"]==dataset]
        for flavor,color in zip(["Online", "Sliding", "Unsupervised", "Historical"],allcolors):
            dftemp = dfdataset[dfdataset["Method"] != "Avg."]
            scores = [perf for perf in dftemp[flavor].values if np.isnan(perf) == False]
            if len(scores)==0:
                continue
            allscores.append(scores)
            allnames.append(flavor)
            colors.append(color)
            print(f"{flavor} : {scores}")
        medians = [np.median(lst) for lst in allscores]

        # Sort data, labels, and colors by median values
        allscores = [lst for _, lst in sorted(zip(medians, allscores))]
        allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
        # Add titles and labels

        box = ax[counter].boxplot(allscores, patch_artist=True)

        # Define colors

        colors = [lbl for _, lbl in sorted(zip(medians, colors))]
        # Color each box
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Add titles and labels
        ax[counter].set_title(dataset)
        #ax[counter].set_xlabel('flavors')
        ax[counter].set_ylabel('AUC (AD1)')
        ax[counter].set_xticks(range(1, len(allnames) + 1), allnames)

    plt.show()

def Methods_per_dataset_all_flavors(df):
    allcolors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', '#800080',
                 '#008080', '#000080']
    fig, ax = plt.subplots(len(df["Dataset"].unique()), 1)
    counter = -1
    plt.title("Method per dataset")
    for dataset in df["Dataset"].unique():
        allscores = []
        allnames = []
        colors = []
        counter += 1
        dfdataset = df[df["Dataset"] == dataset]
        dfdataset = dfdataset[dfdataset["Method"] != "Avg."]
        for tech, color in zip(dfdataset["Method"].unique(), allcolors):
            collect_scores = []
            dftemp_dataset = dfdataset[dfdataset["Method"] == tech]
            for col in ["Online", "Sliding", "Historical", "Unsupervised"]:
                collect_scores.extend([perf for perf in dftemp_dataset[col].values if np.isnan(perf) == False])
            allscores.append(collect_scores)
            allnames.append(tech)
            colors.append(color)
            print(f"{tech} : {collect_scores}")
        medians = [np.median(lst) for lst in allscores]

        # Sort data, labels, and colors by median values
        allscores = [lst for _, lst in sorted(zip(medians, allscores))]
        allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
        # Add titles and labels

        box = ax[counter].boxplot(allscores, patch_artist=True)

        # Define colors

        colors = [lbl for _, lbl in sorted(zip(medians, colors))]
        # Color each box
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Add titles and labels
        ax[counter].set_title(dataset)
        # ax[counter].set_xlabel('flavors')
        ax[counter].set_ylabel('AUC (AD1)')
        ax[counter].set_xticks(range(1, len(allnames) + 1), allnames)

    plt.show()

def kait_test():
    # dominant technqiue
    dftemp=df[df["Dataset"]=="ims "]




    print(dftemp.head())

    from autorank import autorank, plot_stats, create_report, latex_table

    dftemp= dftemp[["Online","Sliding"]]
    df_numeric = dftemp.apply(pd.to_numeric, errors='coerce')

    # Drop rows with any NaN values
    filtered_df = df_numeric.dropna()

    print(filtered_df.head(15))
    result = autorank(filtered_df, alpha=0.05, verbose=False,force_mode="nonparametric")
    create_report(result)
    plot_stats(result,allow_insignificant=True)
    plt.show()

def allMethods_and_flavor_(df):
    allscores = []
    allnames = []
    Bigallnames = []
    Bigallscores = []
    Bigcolors = [
        '#0072B2',  # Strong Blue
        '#009E73',  # Green
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#D55E00',  # Vermilion
        '#F0E442',  # Yellow
        '#CC79A7',  # Reddish Purple
        '#000000',  # Black
        '#999999',  # Gray
        '#F4A582',  # Coral
        '#92C5DE',  # Light Blue
        '#D73027',  # Red
        '#1A9850',  # Forest Green
        '#762A83',  # Dark Purple
        '#A6D96A',  # Light Green
        '#FDAE61',  # Soft Orange
        '#2B83BA',  # Strong Cyan
        '#ABDDA4',  # Light Mint
        '#F46D43',  # Soft Red
        '#3288BD',  # Strong Blue
        '#66C2A5',  # Light Teal
        '#FC8D62',  # Soft Salmon
        '#8DA0CB',  # Muted Blue
        '#E78AC3',  # Pink
        '#A6D854',  # Lime Green
        '#FFD92F',  # Bright Yellow
        '#E5C494',  # Tan
        '#B3B3B3',  # Light Gray
        '#B2DF8A',  # Light Green
        '#FB9A99',  # Soft Pink
        '#1F78B4',  # Strong Blue
        '#33A02C',  # Green
        '#E31A1C',  # Red
        '#FF7F00',  # Orange
        '#6A3D9A',  # Purple
        '#B15928',  # Brown
        '#CAB2D6',  # Lavender
        '#FFFF99',  # Light Yellow
        '#6B6B6B',  # Dark Gray
        '#DA70D6',  # Orchid
        '#8B4513',  # Saddle Brown
        '#20B2AA',  # Light Sea Green
        '#FF6347',  # Tomato
        '#4682B4',  # Steel Blue
        '#9ACD32',  # Yellow Green
        '#DDA0DD',  # Plum
        '#B0C4DE',  # Light Steel Blue
        '#5F9EA0',  # Cadet Blue
        '#F08080',  # Light Coral
    ]
    colors=[]
    counter=-1
    for techname in df["Method"].unique():
        if "Avg" in techname:
            continue
        dftemp = df[df["Method"] == techname]

        for col in ["Online", "Sliding","Historical", "Unsupervised"]:

            collect_scores=[float(perf) if np.isnan(float(perf)) == False else 0 for perf in dftemp[col].values ]
            if sum(collect_scores)==0:
                continue
            Bigallnames.append(f"{techname}_{col[:3]}")
            Bigallscores.append(collect_scores)
    print(len(Bigallnames))
    print(len(Bigcolors))
    allperfbig={}
    for name,score in zip(Bigallnames,Bigallscores):
        allperfbig[name]=score
    barplot_best_per_dataset(allperfbig,df["Method"].unique(),Bigallnames,Bigcolors,df)

def allMethods_and_flavor_across_dataset_without_Historical(df):
    allscores = []
    allnames = []
    Bigallnames = []
    Bigallscores = []
    Bigcolors = [
        '#0072B2',  # Strong Blue
        '#009E73',  # Green
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#D55E00',  # Vermilion
        '#F0E442',  # Yellow
        '#CC79A7',  # Reddish Purple
        '#000000',  # Black
        '#999999',  # Gray
        '#F4A582',  # Coral
        '#92C5DE',  # Light Blue
        '#D73027',  # Red
        '#1A9850',  # Forest Green
        '#762A83',  # Dark Purple
        '#A6D96A',  # Light Green
        '#FDAE61',  # Soft Orange
        '#2B83BA',  # Strong Cyan
        '#ABDDA4',  # Light Mint
        '#F46D43',  # Soft Red
        '#3288BD',  # Strong Blue
        '#66C2A5',  # Light Teal
        '#FC8D62',  # Soft Salmon
        '#8DA0CB',  # Muted Blue
        '#E78AC3',  # Pink
        '#A6D854',  # Lime Green
        '#FFD92F',  # Bright Yellow
        '#E5C494',  # Tan
        '#B3B3B3',  # Light Gray
        '#B2DF8A',  # Light Green
        '#FB9A99',  # Soft Pink
        '#1F78B4',  # Strong Blue
        '#33A02C',  # Green
        '#E31A1C',  # Red
        '#FF7F00',  # Orange
        '#6A3D9A',  # Purple
        '#B15928',  # Brown
        '#CAB2D6',  # Lavender
        '#FFFF99',  # Light Yellow
        '#6B6B6B',  # Dark Gray
    ]
    colors=[]
    counter=-1
    for techname in df["Method"].unique():
        if "Avg" in techname:
            continue
        dftemp = df[df["Method"] == techname]
        techn_best=[0]
        techn_best_name=""

        best_color=Bigcolors[0]
        for col in ["Online", "Sliding", "Unsupervised"]:
            collect_scores=[perf for perf in dftemp[col].values if np.isnan(perf) == False]

            if len(collect_scores)==0:
                continue
            elif np.median(collect_scores)>np.median(techn_best):
                counter+=1
                techn_best=collect_scores
                techn_best_name=f"{techname}_{col[:3]}"
                best_color=Bigcolors[counter]
            else:
                counter+=1
            Bigallnames.append(f"{techname}_{col[:3]}")
            Bigallscores.append(collect_scores)
        allscores.append(techn_best)
        allnames.append(techn_best_name)
        colors.append(best_color)
        print(f"{techn_best_name} : {techn_best}")


    #colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', '#800080',
    #          '#008080', '#000080']


    #critical_diagram(allnames, allscores)
    #boxplots(allnames, allscores, colors)
    #violinplots(allnames, allscores, colors)

    # allperf={}
    # for name,score in zip(allnames,allscores):
    #     allperf[name]=score
    # barplot(allnames,colors,allperf,df)

    allperfbig={}
    for name,score in zip(Bigallnames,Bigallscores):
        allperfbig[name]=score
    #barplot_best_per_dataset(allperfbig,df["Method"].unique(),Bigallnames,Bigcolors,df)


    critical_diagram(Bigallnames, Bigallscores)
    #boxplots(Bigallnames, Bigallscores, Bigcolors)
    violinplots(Bigallnames, Bigallscores, Bigcolors)

def boxplots_text(allnames,allscores,colors,text,labels):
    # Sort data, labels, and colors by median values
    medians = [np.median(lst) for lst in allscores]
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]
    text= [lbl for _, lbl in sorted(zip(medians, text))]
    # Add titles and labels
    fig, ax = plt.subplots()
    medianprops = dict(color="black", linewidth=1.5)
    box = ax.boxplot(allscores, patch_artist=True,medianprops=medianprops)

    # Define colors

    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    #plt.title('Box Plot (choosing the best flavor for Methods)')
    plt.xlabel('Dataset')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames)
    plt.xticks(rotation=45,fontweight='bold')
    # Add text annotations
    for i, txt in enumerate(text):
        # Get the position for the text
        x_position = i + 1  # x-tick positions start at 1
        y_position = np.median(allscores[i]) + 0.01  # Slightly above the max value of the box

        # Add the text annotation
        ax.text(x_position, y_position, txt, ha='center', va='bottom', fontweight='bold')
    import matplotlib.patches as mpatches

    legend_handles = [ mpatches.Patch(color=colorl, label=label) for colorl,label in labels]

    # Add the legend to the plot
    ax.legend(handles=legend_handles, loc='upper left')
    plt.show()


def violinplots_text(allnames, allscores, colors, text, labels):
    import matplotlib.patches as mpatches

    # Sort data, labels, and colors by median values
    medians = [np.median(lst) for lst in allscores]
    sorted_indices = np.argsort(medians)
    allscores = [allscores[i] for i in sorted_indices]
    allnames = [allnames[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    text = [text[i] for i in sorted_indices]

    fig, ax = plt.subplots()

    # Create violin plot
    violin = ax.violinplot(allscores, showmeans=False, showmedians=True)

    # Customize each violin with a different color
    for i, (pc, color) in enumerate(zip(violin['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Set the color for the median lines
    violin['cmedians'].set_linewidth(1.5)
    violin['cmedians'].set_color('black')
    # for line in violin['cmedians']:
    #     line.set_linewidth(1.5)
    #     line.set_color('black')

    # Set the labels and title
    plt.xlabel('Dataset')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames, rotation=45, fontweight='bold')

    # Add text annotations
    for i, txt in enumerate(text):
        x_position = i + 1  # x-tick positions start at 1
        y_position = np.max(allscores[i]) + 0.01  # Slightly above the max value of the violin
        ax.text(x_position, y_position, txt, ha='center', va='bottom', fontweight='bold')
    plt.ylim([0,1.2])
    # Add a legend
    legend_handles = [mpatches.Patch(color=colorl, label=label) for colorl, label in labels]
    ax.legend(handles=legend_handles, loc='upper left')

    plt.show()
def boxplots_ax(allnames,allscores,colors,ax):
    medians = [np.median(lst) for lst in allscores]
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]

    # Add titles and labels
    medianprops = dict(color="black", linewidth=1.5)
    box = ax.boxplot(allscores, patch_artist=True, medianprops=medianprops)

    # Define colors

    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    # plt.title('Box Plot (choosing the best flavor for Methods)')
    ax.set_xticks(range(1, len(allnames) + 1), allnames,rotation=0, fontweight='bold')
def boxplots(allnames,allscores,colors):
    # Sort data, labels, and colors by median values
    medians = [np.median(lst) for lst in allscores]
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]

    # Add titles and labels
    fig, ax = plt.subplots()
    medianprops = dict(color="black", linewidth=1.5)
    box = ax.boxplot(allscores, patch_artist=True,medianprops=medianprops)

    # Define colors

    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    #plt.title('Box Plot (choosing the best flavor for Methods)')
    plt.xlabel('Methods')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames)
    plt.xticks(rotation=90,fontweight='bold')

    plt.show()
def violinplots(allnames, allscores, colors):
    import matplotlib.patches as mpatches

    # Sort data, labels, and colors by median values
    medians = [np.median(lst) for lst in allscores]
    sorted_indices = np.argsort(medians)
    allscores = [allscores[i] for i in sorted_indices]
    allnames = [allnames[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]

    fig, ax = plt.subplots()

    # Create violin plot
    violin = ax.violinplot(allscores, showmeans=False, showmedians=True)

    # Customize each violin with a different color
    for i, (pc, color) in enumerate(zip(violin['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Set the color for the median lines
    violin['cmedians'].set_linewidth(1.5)
    violin['cmedians'].set_color('black')
    # for line in violin['cmedians']:
    #     line.set_linewidth(1.5)
    #     line.set_color('black')

    # Set the labels and title
    plt.xlabel('Methods')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames)
    plt.xticks(rotation=90,fontweight='bold')


    plt.show()
def critical_diagram(allnames, allscores):
    allnames=[name.lower() for name in allnames]
    dfcd = pd.DataFrame()
    for name, scores in zip(allnames, allscores):
        print(name)
        dfcd[name] = scores
    print(dfcd.head(5))
    result = autorank(dfcd, alpha=0.05, verbose=False, force_mode="nonparametric")
    create_report(result)
    plot_stats(result, allow_insignificant=True)
    plt.show()




def best_choice_per_flavor_across_dataset(df):
    allperf={}
    allperf["Online"]=[]
    flavors= ["Online", "Sliding","Unsupervised"]
    allperf["Sliding"]=[]
    allperf["Unsupervised"]=[]
    for dataset in df["Dataset"].unique():
        dftemp=df[df["Dataset"]==dataset]
        for col in flavors:
            best=max([perf for perf in dftemp[col].values if np.isnan(perf)==False])
            allperf[col].append(best)

    allnames=[]
    allscores=[]
    for key in allperf.keys():
        allnames.append(key)
        allscores.append(allperf[key])

    medians = [np.median(lst) for lst in allscores]

    # Sort data, labels, and colors by median values
    allscores = [lst for _, lst in sorted(zip(medians, allscores))]
    allnames = [lbl for _, lbl in sorted(zip(medians, allnames))]
    # Add titles and labels
    fig, ax = plt.subplots()
    box = ax.boxplot(allscores, patch_artist=True)

    # Define colors
    colors = ['#FF0000', '#00FF00', '#0000FF']
    colors = [lbl for _, lbl in sorted(zip(medians, colors))]
    # Color each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add titles and labels
    plt.title('Box Plot flavors across all datasets')
    plt.xlabel('flavors')
    plt.ylabel('AUC (AD1)')
    plt.xticks(range(1, len(allnames) + 1), allnames)

    plt.show()

    critical_diagram(allnames, allscores)

    # this is maybe not right.

    barplot(flavors,colors,allperf,df)
def barplot(flavors,colors,allperf,df):
    datasets = df["Dataset"].unique()
    fig, ax = plt.subplots(10)
    # colors = sns.color_palette("husl", len(flavors))
    color_dict = dict(zip(flavors, colors))
    for i in range(len(datasets)):
        dfbar = pd.DataFrame()
        dfbar[" "] = flavors
        dfbar["AUC\n(AD1)"] = [allperf[key][i] for key in allperf]
        dfbar = dfbar.sort_values(by='AUC\n(AD1)', ascending=False)
        # Create bar plot
        sns.barplot(x=' ', y='AUC\n(AD1)', data=dfbar, ax=ax[i], palette=color_dict)
        #ax[i].set_title(datasets[i])
        for q in range(len(ax[i].containers)):
            ax[i].containers[q][0].set_alpha(0.6)
        ax[i].set_ylim(0, 1.6)
        for p,name in zip(ax[i].patches,dfbar[" "].values):

            ax[i].annotate(f"{p.get_height():.2f}",
                           (p.get_x()+ p.get_width()/2, 1.0),
                           xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold')
            ax[i].annotate(name.lower().replace("_","\n"),(p.get_x() + p.get_width()/3, 0.1),
                           textcoords='offset points', ha='center',
                           rotation=0, fontsize=11, fontweight='bold', color='black')
        ax[i].set_xticks([])
        ax[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[i].set_ylabel(datasets[i])
    plt.show()


def barplot_best_per_dataset(dict_res,uniq_technames,Bigtechnames,colors,df):
    datasets = df["Dataset"].unique()
    fig, ax = plt.subplots(10)
    # colors = sns.color_palette("husl", len(flavors))
    color_dict = dict(zip(Bigtechnames, colors))
    for i in range(len(datasets)):
        allperf= {}
        for tech in uniq_technames:
            # finde best flavor
            maxx=-np.inf
            maxname="none"
            for keyy in dict_res.keys():
                if f"{tech}_" in keyy:
                    if maxx<dict_res[keyy][i]:
                        maxx=dict_res[keyy][i]
                        maxname=keyy
            allperf[maxname]=maxx

        dfbar = pd.DataFrame()
        dfbar[" "] = [key for key in allperf]
        dfbar["AUC\n(AD1)"] = [allperf[key] for key in allperf]
        dfbar = dfbar.sort_values(by='AUC\n(AD1)', ascending=True)
        # Create bar plot
        sns.barplot(x=' ', y='AUC\n(AD1)', data=dfbar, ax=ax[i], palette=color_dict)
        #ax[i].set_title(datasets[i])
        for q in range(len(ax[i].containers)):
            ax[i].containers[q][0].set_alpha(0.6)
        ax[i].set_ylim(0, 1.6)
        for p,name in zip(ax[i].patches,dfbar[" "].values):

            ax[i].annotate(f"{p.get_height():.2f}",
                           (p.get_x()+ p.get_width()/2, 1.0),
                           xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold')
            ax[i].annotate(name.lower().replace("_","\n"),(p.get_x() + p.get_width()/3, 0.1),
                           textcoords='offset points', ha='center',
                           rotation=0, fontsize=11, fontweight='bold', color='black')
        ax[i].set_xticks([])
        ax[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[i].set_ylabel(datasets[i])
    plt.show()

def allMethods_and_flavor_across_dataset_with_Historic(df):
    allscores = []
    allnames = []
    Bigallnames = []
    Bigallscores = []
    Bigcolors = [
        '#FF0000',  # Red
        '#00FF00',  # Green
        '#0000FF',  # Blue
        '#FFFF00',  # Yellow
        '#FF00FF',  # Magenta
        '#00FFFF',  # Cyan
        '#800000',  # Maroon
        '#808000',  # Olive
        '#800080',  # Purple
        '#008080',  # Teal
        '#000080',  # Navy
        '#FFA500',  # Orange
        '#A52A2A',  # Brown
        '#7FFF00',  # Chartreuse
        '#D2691E',  # Chocolate
        '#FF4500',  # OrangeRed
        '#2E8B57',  # SeaGreen
        '#4682B4',  # SteelBlue
        '#8B008B',  # DarkMagenta
        '#556B2F',  # DarkOliveGreen
        '#8B4513',  # SaddleBrown
        '#9932CC',  # DarkOrchid
        '#FF6347',  # Tomato
        '#00CED1',  # DarkTurquoise
        '#FFD700',  # Gold
        '#ADFF2F',  # GreenYellow
        '#F08080',  # LightCoral
        '#20B2AA',  # LightSeaGreen
        '#87CEFA',  # LightSkyBlue
        '#778899',  # LightSlateGray
        '#B0C4DE',  # LightSteelBlue
        '#32CD32',  # LimeGreen
        '#191970',  # MidnightBlue
        '#FFE4B5'  # Moccasin
    ]
    colors = []
    counter = -1

    datasets_wit_historic=[]
    for dataset in df["Dataset"]:
        historicv= df[df["Dataset"]==dataset]["Historical"].values
        if sum([1 for v in historicv if np.isnan(v)])==len(historicv):
            continue
        datasets_wit_historic.append(dataset)
    df=df[df["Dataset"].isin(datasets_wit_historic)]


    for techname in df["Method"].unique():
        if "Avg" in techname:
            continue
        dftemp = df[df["Method"] == techname]
        techn_best = [0]
        techn_best_name = ""

        best_color = Bigcolors[0]
        for col in ["Online", "Sliding","Historical", "Unsupervised"]:
            collect_scores = [perf for perf in dftemp[col].values if np.isnan(perf) == False]

            if len(collect_scores) == 0:
                continue
            elif np.median(collect_scores) > np.median(techn_best):
                counter += 1
                techn_best = collect_scores
                techn_best_name = f"{techname}_{col[:3]}"
                best_color = Bigcolors[counter]
            else:
                counter += 1
            Bigallnames.append(f"{techname}_{col[:3]}")
            Bigallscores.append(collect_scores)
        allscores.append(techn_best)
        allnames.append(techn_best_name)
        colors.append(best_color)
        print(f"{techn_best_name} : {techn_best}")

    # colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', '#800080',
    #          '#008080', '#000080']
    if len(allscores[0])>=5:
        critical_diagram(allnames, allscores)
    boxplots(allnames, allscores, colors)

    allperf = {}
    for name, score in zip(allnames, allscores):
        allperf[name] = score
    barplot(allnames, colors, allperf,df)
    if len(allscores[0]) >= 5:
        critical_diagram(Bigallnames, Bigallscores)
    boxplots(Bigallnames, Bigallscores, Bigcolors)


def easy_and_hard_cases():
    synthetic = ["cmapss", "azure"]
    experimental = ["ims", "femto","xjtu"]
    real = ["navarchos", "edp", "metro","bhd", "formula1"]
    type_colors=['#0072B2', '#E69F00', '#009E73']
    label=["synthetic","experimental","real"]
    names=[
        "cmapss",
        "femto",
        "ims",
        "edp",
        "metro",
        "navarchos",
        "azure",
        "bhd",
        "xjtu",
        "formula1",
        "cnc"
    ]
    ratio=[
        (13*709)/160359, # "cmapss",
        (6*52)/7534, #         "femto",
        (99*3)/9464, #         "ims",
        (8640*8)/209236, #         "edp",
        (4*288)/25521, #         "metro",
        (30414)/854178, #         "navarchos", number of points in PH obtained from evaluation module
        (761*96)/876100, #         "azure",
        (97*10)/146654, #         "bhd",
        (15*18)/9216, #         "xjtu",
        (465325)/5563727, #         "formula1", number of points in PH obtained from evaluation module
        (14*15)/6304, # cnc
    ]
    totext=[]
    allscores = []
    allnames = []
    colors=[]
    for dataset in names:
        print((f"{dataset} PHr:\n{str(ratio[names.index(dataset.lower())])[:5]}"))
    # dfn=df[df["Method"]!="Avg."]
    # dfn=dfn[dfn["Method"]!="all"]
    # dfn=dfn[dfn["Method"]!="add"]
    # for dataset in dfn["Dataset"].unique():
    #     dftemp = dfn[dfn["Dataset"]==dataset]
    #     totext.append(f"PHr:\n{str(ratio[names.index(dataset.lower())])[:5]}")
    #     scores = []
    #     for col in ["Online", "Sliding", "Historical", "Unsupervised"]:
    #         scores.extend([perf for perf in dftemp[col].values if np.isnan(perf) == False])
    #     if dataset.lower() in synthetic or "AZUR" in dataset:
    #         print(dataset.lower())
    #         colors.append(type_colors[0])
    #     elif dataset.lower() in experimental:
    #         colors.append(type_colors[1])
    #     else:
    #         colors.append(type_colors[2])
    #     allscores.append(scores)
    #     allnames.append(dataset)
        #print(f"{dataset} : {scores}")

    # colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', '#800080',
    #          '#008080']
    # boxplots_text(allnames, allscores, colors,totext,zip(type_colors,label))
    # violinplots_text(allnames, allscores, colors,totext,zip(type_colors,label))

def per_Method_analyses(df):
    flavors = ["Online", "Incr", "Unsu","Hist"]
    fcolors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
    df = df[df["Method"] != "Avg."]
    teckname=[]
    tech_flavors_name=[]
    teck_flavors_scores=[]
    for tech in df["Method"].unique():
        flavor_names=[]
        flavor_scores=[]
        dftemp = df[df["Method"] == tech]
        for col in ["Online", "Sliding", "Unsupervised"]:
            if dftemp[col].isna().any():
                continue
            flavor_names.append(col[:4])
            flavor_scores.append([perf for perf in dftemp[col].values if np.isnan(perf) == False])
        if len(flavor_names)<2:
            continue
        teckname.append(tech)
        tech_flavors_name.append(flavor_names)
        teck_flavors_scores.append(flavor_scores)

    fig,ax =plt.subplots(1,len(teckname))
    print(len(teckname))
    for i,tech,flavor_names,flavor_scores in zip(range(len(teckname)),teckname,tech_flavors_name,teck_flavors_scores):
        colors=[fcolors[flavors.index(fn)] for fn in flavor_names]
        boxplots_ax(flavor_names,flavor_scores,colors,ax[i])
        ax[i].set_title(tech)
    plt.show()

def per_Method_analyses_Historic(df):
    df = df.dropna(subset=["Historical"])
    flavors = ["Onli", "Slid", "Unsu","Hist"]
    fcolors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
    df = df[df["Method"] != "Avg."]
    teckname=[]
    tech_flavors_name=[]
    teck_flavors_scores=[]
    for tech in df["Method"].unique():
        flavor_names=[]
        flavor_scores=[]
        dftemp = df[df["Method"] == tech]
        for col in ["Online", "Sliding", "Unsupervised","Historical"]:
            if dftemp[col].isna().any():
                continue
            flavor_names.append(col[:4])
            flavor_scores.append([perf for perf in dftemp[col].values if np.isnan(perf) == False])
        if len(flavor_names)<2:
            continue
        teckname.append(tech)
        tech_flavors_name.append(flavor_names)
        teck_flavors_scores.append(flavor_scores)

    fig,ax =plt.subplots(1,len(teckname))
    print(len(teckname))
    for i,tech,flavor_names,flavor_scores in zip(range(len(teckname)),teckname,tech_flavors_name,teck_flavors_scores):
        colors=[fcolors[flavors.index(fn)] for fn in flavor_names]
        boxplots_ax(flavor_names,flavor_scores,colors,ax[i])
        ax[i].set_title(tech)
    plt.show()

#def synthetic_vs_experimental_vs




# df = pd.read_csv("results_AUC_AD1.txt",sep=",",index_col=None)



# print(df.head())

# correct:
# allMethods_and_flavor_(df)
# allMethods_and_flavor_across_dataset_without_Historical(df)
#allMethods_and_flavor_across_dataset_with_Historic(df)
#best_technque_per_flavor()

easy_and_hard_cases()

# generic flavors impact
#flavors_all_datasets_on_common_tecn(df)
#flavors_all_datasets_on_common_tecn_semi_Online_Sliding(df)
#flavors_all_datasets_on_common_tecn_semi_Online_Sliding_historic(df)
#flavors_all_datasets_on_common_tecn_semi_Online_Sliding_historic_and_Usupervised(df)

#per_Method_analyses(df)
#per_Method_analyses_Historic(df)











############################################################3333333
#best_choice_per_flavor_across_dataset(df)
# flavors


#allMethods_and_flavor_best(df)
#all_flavors_with_and_without_historic()
#flavors_all_datasets(df)


#allDatasetsFlavorsAvg(df)
#Methods_all_datasets_all_flavors(df)
#Methods_all_datasets_best_flavor(df)
#Methods_per_dataset_all_flavors(df)
#flavors_all_datasets(df)
#flavors_per_dataset(df)
#SAND : [0.884, 0.996, 0.853, 0.52, 0.823]
