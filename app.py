import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import codecs
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sweetviz as sv
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

import time

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from random import randint

state = randint(0, 2023)

st.set_option('deprecation.showPyplotGlobalUse', False)

import streamlit as st
from PIL import Image

image = Image.open('./Dataset.png')


def st_display_sweetviz(report_html,width=1000,height=500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page,width=width,height=height,scrolling=True)

def delay_screen(args:str, delay_time:float = 0.01):
    # To introduce delay
    progress_text = f"{args} in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(delay_time)
        my_bar.progress(percent_complete + 1, text=progress_text)
    # time.sleep(1)
    # my_bar.empty()

def preprocess_dataset(df):
    # Extracting the title from the name
    df["Title"] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0])
    
    # Grouping people together under PClass and Title
    df.groupby(['Pclass', 'Title']).agg({'Age':'mean'}).round(0)
    
    # Assigning the mean of the Age of each group to the respective missing values 
    df['Age'] = df.groupby(['Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.mean())) 
    # comment to understand why this step is important
    
    # Removing redundant features (search column wise and removing particular rows )
    df = df.drop(columns = ['Name', 'Ticket', 'PassengerId','Cabin', 'Title'])
    df = df.dropna(subset = ['Fare' ,'Embarked'])
    
    # Encoding all categorical features into numerals
    df.Sex = pd.Categorical(df.Sex)
    df.Embarked = pd.Categorical(df.Embarked)
    df["Sex"] = df.Sex.cat.codes
    df["Embarked"] = df.Embarked.cat.codes
    return df
    
def app(title=None):
    st.title(title)

    st.subheader("Let's first understand the Dataset Glossary")
    st.image(image, caption='Data Glossary')

    st.subheader("Quick Preview")
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')
    st.write(train_df.head(10))
    
    st.subheader("Quick Analysis")
    # Use the analysis function from sweetviz module to create a 'DataframeReport' object.
    feature_config = sv.FeatureConfig(skip="PassengerId", force_text=["Age"])
    my_report = sv.compare([train_df, "Training Data"], [test_df, "Test Data"], "Survived", feature_config)
    
    # Uncomment to get single analysis of Training Data
    # analysis = sv.analyze([train,'cegis'], 
    #                       feat_cfg= sv.FeatureConfig(skip="PassengerId", 
    #                                                 force_text=["Age"]),
                            # target_feat=None)
    
    my_report.show_html(filepath='./cegis.html', open_browser=False, layout='vertical', scale=1.0) # Renders the output on a web page but Default arguments will generate at "SWEETVIZ_REPORT.html" path name
    st_display_sweetviz('./cegis.html')
    
    st.subheader("Why should we conduct Statistical Tests in the first place?")
    st.write("Hypothesis-Did the rich people on the Titanic had a higher survival rate than the others based on PClass or not? Approach-The data has several features but we are only concerned about the followings:\n - Survived-A category of either 0 or 1 which indicates whether that individual survived.\n - Pclass-The ship was divided into three classes. First, Second, and Third.\n - Fare-The price that this individual paid for the ticket.")
    # Distribution for rich:
    first_fares = train_df["Fare"][train_df["Pclass"]==1]
    first_mean = round(np.mean(first_fares), 2)
    first_median = round(np.median(first_fares), 2)
    first_conf = np.round(np.percentile(first_fares, [2.5, 97.5]), 2)

    # Distribution for Poor
    third_fares = train_df["Fare"][train_df["Pclass"]==3]
    third_mean = round(np.mean(third_fares), 2)
    third_median = round(np.median(third_fares), 2)
    third_conf = np.round(np.percentile(third_fares, [2.5, 97.5]), 2)
    
    plot_container = st.container()
    
    with plot_container:
        #plot1, plot2 = st.columns(1
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 7))
        ax1.hist(first_fares, color = 'lightyellow', ec = 'red')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.70, 0.98, f"Mean: {first_mean} \nMedian: {first_median} \nCI: {first_conf}",transform=ax1.transAxes ,fontsize=14,
                    verticalalignment='top', bbox=props)
        ax1.set_xlabel("Fare")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of the fare of the tickets in the first class")
            
        ax2.hist(third_fares, color = 'lightblue', ec = 'red')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.72, 0.98, f"Mean: {third_mean} \nMedian: {third_median} \nCI: {third_conf}",transform=ax2.transAxes ,fontsize=14,
                    verticalalignment='top', bbox=props)
        ax2.set_xlabel("Fare")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of the fare of the tickets in the third class")
        st.pyplot(fig1)
        
        st.write("To be extra careful, we will conduct hypothesis testing which is used to check if the observed difference between the two populations is really significant or is just due to some randomness/bias in the data. Considering there are many kinds of methods to achieve the above hence we will resort to the most widely used i.e. Z-test. \n\n Note: In scientific terms, that assumption can be framed using the concept of Null and Alternative hypothesis:\n- Null Hypothesis: The socio-economic class of the people didn’t have an effect on the survival rate.\n- Alternative Hypothesis: The socio-economic class of the people affected their survival rate.\n\n**The test would is being conducted to see if the null hypothesis should be rejected or accepted!!**")
        
        with st.expander("**Main Reason**"):
            st.write("The probability distribution of the survival rate of the first-class people and the probability distribution of the survival rate for the third-class both have to be normally distributed. However, we don’t really have the data of the entire population, we just have a sample population. The fix to this problem is the central limit theorem. If we take a large enough sample of means from the population, then our sample distribution is going to be normally distributed.") 
            
            First_Class_Sample = np.array([np.mean(train_df[train_df["Pclass"]==1].sample(20)["Survived"].values) for i in range(100)])
            Third_Class_Sample = np.array([np.mean(train_df[train_df["Pclass"]==3].sample(20)["Survived"].values) for i in range(100)])

            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 7))
            
            sns.set(style='dark')
            sns.distplot(First_Class_Sample, ax = ax1, color = 'red')
            
            ax1.set_title("First-Class Sample Distribution")
            ax1.set_xlabel("Survival Rate")
            ax1.set_ylabel("Frequency")
           
            sns.distplot(Third_Class_Sample, ax = ax2)
            ax2.set_title("Third-Class Sample Distribution")
            ax2.set_xlabel("Survival Rate")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)
            plt.clf()
            
        
        
        #delay_screen(args = 'Statistical Testing Results', delay_time = 0.1)
        #with st.spinner('Statistical Testing Results being evaluated....'):
            #time.sleep(10)
       # We first calculated the Z-score and then the P_value referenced at 0.05
            effect = np.mean(First_Class_Sample) - np.mean(Third_Class_Sample)
            sigma_first = np.std(First_Class_Sample)
            sigma_third = np.std(Third_Class_Sample)
            sigma_difference = np.sqrt((sigma_first**2)/len(First_Class_Sample)+(sigma_third**2)/len(Third_Class_Sample))
            z_score = effect / sigma_difference

            col1, col2 = st.columns(2)
            col2.metric(label = 'Z-Score',value = f"{z_score.round(3)}")

            p_value = stats.norm.sf(abs(z_score))*2 # since it is a two tailed test we are conducting
            col1.metric(label = 'P-Value',value = f"{p_value}")


        
     # Preprocessing and Data Cleaning
    train_df = preprocess_dataset(df=train_df)
    
    # Dropping our target variable
    y = train_df.Survived.values
    train_df = train_df.drop(columns =["Survived"])
    
    
    # Train test split from original dataset
    x_train, x_test, y_train, y_test = train_test_split(train_df, y, test_size=0.1, random_state=state)
    no = randint(0, len(x_test))

    option = st.selectbox(
        'Which Model Do you wanna use to conduct training?',
        ('Choose from below','Logistic Regression','Random Forests'),
        index = None,
        placeholder = "Available Model Types..."
    )

    if option == 'Logistic Regression':
        
        # Model Initialization and Training
        model = LogisticRegression()

        if st.button("Train", type="primary"):
            
            #Calling delay screen function to make loading widget
            delay_screen(args = 'Training') 
            model.fit(x_train, y_train) 
            pred=model.predict(x_test)
            # st.write(f'Accuracy Score: {(model.score(x_test, y_test)*100).round(2)}%') # Accuracy Score
            #st.metric(label = 'Accuracy',value = f"{(model.score(x_test, y_test)*100).round(2)}%")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label = 'Accuracy',value = f"{(accuracy_score(y_test,pred)*100).round(3)}%")
            col2.metric(label = 'Recall',value = f"{(recall_score(y_test,pred)*100).round(3)}%")
            col3.metric(label = 'Precision',value = f"{(precision_score(y_test,pred)*100).round(3)}%")
            col4.metric(label = 'F1-Score',value = f"{(f1_score(y_test,pred)*100).round(3)}%")
          
            # Shap Values
            time.sleep(2)
            st.write("Let's see whether our trained models are really blackboxes or not???")
            delay_screen(args = 'Interpreting', delay_time = 0.02)
            explainer = shap.Explainer(model, x_train)
            shap_values = explainer(x_test)
            st.pyplot(shap.summary_plot(shap_values, x_test, color=plt.get_cmap("cool")), clear_figure = True)
            with st.expander("**Did you know about this Graph??**"):
                st.write("Its a ***Beeswarm Plot***.\n- High values of the Latitude variable have a high negative contribution on the prediction, while low values have a high positive contribution.\n- All variables are shown in the order of global feature importance, the first one being the most important and the last being the least important one.")
            # st.pyplot(fig = shap.plots.bar(shap_values[0]))
            
            st.subheader("Can we match above interpretation on the basis of a single test case???")
            #no = 1
            st.write(x_test.iloc[no])
            if model.predict(np.expand_dims(x_test.iloc[no],axis=0))[0] == 1:
                st.write("The passenger survived")
            else:
                st.write("The passenger did not survive")
            st.pyplot(shap.plots.waterfall(shap_values[1]))
            
            with st.expander("**What is a Waterfall Plot, you ask??**"):
                st.write("In the waterfall plot above, the x-axis has the values of the target (dependent) variable which is the survival or not (0-1). **X** is the chosen observation, **f(x)** is the predicted value of the model, given input **X** and **E[f(x)** is the expected value of the target variable, or in other words, the mean of all predictions *(mean(model.predict(X)))*.\n -  The sum of all SHAP values in the graph above will be equal to E[f(x)] — f(x). \n -  The absolute of SHAP values present above shows us how much a single feature affected the prediction")
            st.pyplot(shap.plots.force(shap_values[1], matplotlib=True))  
                        
    elif option == 'Random Forests':
        
        # Model Initialization and Training
        model_rf=RandomForestClassifier(random_state=state)
        param_grid = { 
            'n_estimators': [150,180],
            'max_depth' : [5,6,7],
            'criterion' :['gini', 'entropy'],
            'min_samples_split' : [2,3,4] }
        CV_rfc = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv= 5)
        
        if st.button("Train", type="primary"):
            delay_screen(args = 'Training', delay_time = 0.001)
            CV_rfc.fit(x_train, y_train)
            model=RandomForestClassifier(random_state=state, 
                                           n_estimators=CV_rfc.best_params_['n_estimators'],
                                           max_depth=CV_rfc.best_params_['max_depth'], 
                                           criterion=CV_rfc.best_params_['criterion'],
                                           min_samples_split=CV_rfc.best_params_['min_samples_split'])    
            model.fit(x_train, y_train)
            pred=model.predict(x_test)
            #print(f"Accuracy for Random Forests: {(accuracy_score(y_test,pred)*100).round(2)}%")
            #st.metric(label = 'Accuracy',value = f"{(accuracy_score(y_test,pred)*100).round(2)}%")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label = 'Accuracy',value = f"{(accuracy_score(y_test,pred)*100).round(3)}%")
            col2.metric(label = 'Recall',value = f"{(recall_score(y_test,pred)*100).round(3)}%")
            col3.metric(label = 'Precision',value = f"{(precision_score(y_test,pred)*100).round(3)}%")
            col4.metric(label = 'F1-Score',value = f"{(f1_score(y_test,pred)*100).round(3)}%")
          
            time.sleep(2)
            
            # Shap Values
            st.write("## Let's see whether our trained models are really blackboxes or not???")
            delay_screen(args = 'Training', delay_time = 0.001)
            explainer = shap.Explainer(model, x_train)
            shap_values = explainer(x_test)

            st.pyplot(shap.summary_plot(shap_values[:,:,1], x_train, plot_type = 'bar'), clear_figure= True)
            
            st.pyplot(shap.summary_plot(shap_values[:,:,1], x_test))
            with st.expander("**Did you know about this Graph??**"):
                st.write("Its a Beeswarm Plot.\n- All variables are shown in the order of global feature importance, the first one being the most important and the last being the least important one.")

            # change the number for a different passenger analysis
            #no = 1
            st.subheader("Can we match above interpretation with a single test case???")
            st.write(x_test.iloc[no])
            if model.predict(np.expand_dims(x_test.iloc[no],axis=0))[0] == 1:
                st.write("The passenger survived")
            else:
                st.write("The passenger did not survive")
            
            st.pyplot(fig = shap.plots.waterfall(shap_values[no,:,1]))
            with st.expander("**What is a Waterfall Plot, you ask??**"):
                st.write("In the waterfall plot above, the x-axis has the values of the target (dependent) variable which is the survival or not (0-1). **X** is the chosen observation, **f(x)** is the predicted value of the model, given input **X** and **E[f(x)** is the expected value of the target variable, or in other words, the mean of all predictions *(mean(model.predict(X)))*.\n -  The sum of all SHAP values in the graph above will be equal to E[f(x)] — f(x). \n -  The absolute of SHAP values present above shows us how much a single feature affected the prediction")
            st.pyplot(fig = shap.plots.force(shap_values[no,:,1], matplotlib=True))

app(title='How end-to-end ML Journey 🤔 Looks like?? ')
