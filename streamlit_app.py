import streamlit as st
import pandas as pd
from sklearn import datasets
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score, f1_score, precision_score
from sklearn import ensemble
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve,auc
import numpy as np
import matplotlib.pyplot as plt
import os



data=datasets.load_iris()
iris=pd.DataFrame(data.data,columns=data.feature_names)
iris["target"]=data.target
iris["target_names"]=iris["target"].replace({0:'setosa',1:'versicolor',2:'virginica'})



def format(fig):
    fig.update_yaxes(matches=None, showticklabels=True, visible=True)
    fig.update_annotations(font=dict(size=16))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
        if type(fig.layout[axis]) == go.layout.XAxis:
            fig.layout[axis].title.text = ''

st.set_page_config(
    page_title='鸢尾花',
    page_icon=' ',
    layout='wide'
)

with st.sidebar:
    st.title('欢迎来到我的应用')
    st.markdown('---')
    # st.markdown('这是它的特性：\n- 数据可视化\n- 模型')

st.experimental_feature_warning('user_info_proxy')
# 获取当前用户信息
user_info = st.experimental_user_info_proxy.get_current_user_info()
# 显示用户信息
st.write('db_username:', user_info['db_username'])
st.write('db_password:', user_info['db_password'])


st.write("Has environment variables been set:",
         os.environ["db_username"] == st.secrets["db_username"],
         os.environ["db_password"] == st.secrets["db_password"])




def page_Visualization():
    st.title("鸢尾花数据可视化")
    st.info('不同物种各个特征的小提琴图')
    ### 数据可视化
    fig1 = go.Figure()
    for item,color in zip(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
        'petal width (cm)'],['violet','tomato','yellowgreen','slateblue']):
        
        fig1.add_trace(go.Violin(
            x=iris['target_names'],
            y=iris[item],
            name=item,
            box_visible=True,
            meanline_visible=True,
            line_color='snow', 
            opacity=0.7,
            fillcolor=color,
            y0=item,
            showlegend=True,
        ))
        fig1.update_layout(violinmode='group')
    format(fig1)
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

    st.info('不同物种各个特征的相关性热图')


    
    corr=iris[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
        'petal width (cm)']].corr().apply(lambda x:round(x,2))

    index=corr.index.tolist()
    columns=corr.columns.tolist()

    # data1=[]
    # for i in index:
    #     data2=[]
    #     for j in columns:
    #         data2.append(corr[i,j])
    #     data1.append(data2)

    x=index
    y=columns
    z =corr.values
    z_text =corr.values

    fig = ff.create_annotated_heatmap(
        z,  
        x=x,
        y=y,
        annotation_text=z_text,
        colorscale="sunset",
        showscale=True
    )
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size=12

    format(fig)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

### 建模
def train_data(x_train,x_test,y_train,y_test):
    def try_different_method(clf,name):
        clf.fit(x_train,y_train)
        result = clf.predict(x_test)
        f1 = f1_score(y_test,result,average='macro')
        accuracy = accuracy_score(y_test,result)
        recall=recall_score(y_test,result,average='macro')
        precision=precision_score(y_test,result,average='macro')
        return f1,accuracy,recall,precision

    rf =ensemble.RandomForestClassifier(max_depth=1,random_state=42,min_samples_split=10)
    rf_socre = try_different_method(rf,'RandomForestClassifier')

    dtr = DecisionTreeClassifier(max_depth=3,random_state=42,min_samples_split=10)
    dtr_score = try_different_method(dtr,'DecisionTreeClassifier')

    lgb = LGBMClassifier(max_depth=3,verbose=-1,random_state=42)
    lgb_score = try_different_method(lgb,'LGBMClassifier')

    xgb = XGBClassifier(max_depth=3,random_state=42,min_samples_split=10)
    xgb_score = try_different_method(xgb,'XGBMClassifier')
    

    result=pd.DataFrame({'模型\指标':["f1","accuracy","recall","precision"],
                        'RandomForestClassifier':rf_socre,
                            'DecisionTreeClassifier':dtr_score,
                            'LGBMClassifier':lgb_score,
                            'XGBMClassifier':xgb_score})

    result=result.T.reset_index()
    result.columns=result.iloc[0,:].tolist()
    result=result.iloc[1:,:]
    result=result.sort_values(["accuracy","f1"],ascending=[False,False]).reset_index(drop=True)

    return result    

def page_model():
    st.title('运用模型')
    x=iris[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
    y=iris[["target"]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.6,random_state=42)
    st.info("基础模型计算结果")
    result=train_data(x_train,x_test,y_train,y_test)
    st.dataframe(result)
    st.info("基于最优模型进一步训练优化")
    clf=ensemble.RandomForestClassifier(max_depth=3,random_state=42,min_samples_split=10)
    clf.fit(x_train,y_train)
    result = clf.predict(x_test)
    f1 = f1_score(y_test,result,average='macro')
    accuracy = accuracy_score(y_test,result)
    recall=recall_score(y_test,result,average='macro')
    precision=precision_score(y_test,result,average='macro')
    st.markdown(":blue[*优化模型后模型指标*]")
    result=pd.DataFrame({'模型指标':["f1","accuracy","recall","precision"],
                         '取值':[f1,accuracy,recall,precision]})
    st.dataframe(result)

    df = pd.DataFrame()
    pre_score = clf.predict_proba(x_test)
    df['y_test'] = y_test

    df['pre_score1'] = pre_score[:,0]
    df['pre_score2'] = pre_score[:,1]
    df['pre_score3'] = pre_score[:,2]

    pre1 = df['pre_score1']
    pre1 = np.array(pre1)

    pre2 = df['pre_score2']
    pre2 = np.array(pre2)

    pre3 = df['pre_score3']
    pre3 = np.array(pre3)

    y_list = df['y_test'].to_list()
    pre_list=[pre1,pre2,pre3]

    lable_names=['0','1','2']
    colors1 = ["r","b","g"]
    colors2 = ["mistyrose","skyblue","palegreen"]
    my_list = []
    linestyles =['solid', 'dashdot', 'dashed']

    plt.figure(figsize=(12,5))

    fig,ax=plt.subplots()
    #plt.figure(figsize=(12,5),facecolor='w')
    for i in range(3):
        roc_auc = 0
        if i==0:
            fpr, tpr, threshold = roc_curve(y_list,pre_list[i],pos_label=0)
            # 计算AUC的值
            roc_auc = auc(fpr, tpr)
            ax.text(0.3, 0.01, "class "+lable_names[i]+' :ROC curve (area = %0.2f)' % roc_auc)
        elif i==1:
            fpr, tpr, threshold = roc_curve(y_list,pre_list[i],pos_label=1)
            # 计算AUC的值
            roc_auc = auc(fpr, tpr)
            ax.text(0.3, 0.11, "class "+lable_names[i]+' :ROC curve (area = %0.2f)' % roc_auc)
        elif i==2:
            fpr, tpr, threshold = roc_curve(y_list,pre_list[i],pos_label=2)
            # 计算AUC的值
            roc_auc = auc(fpr, tpr)
            ax.text(0.3, 0.21, "class "+lable_names[i]+' :ROC curve (area = %0.2f)' % roc_auc)
        my_list.append(roc_auc)
        # 添加ROC曲线的轮廓
        ax.plot(fpr, tpr, color = colors1[i],linestyle = linestyles[i],linewidth = 3,label = "class:"+lable_names[i])  #  lw = 1,
        #绘制面积图
        ax.stackplot(fpr, tpr,alpha = 0.5,edgecolor = colors1[i],colors=colors2)
    
    # 添加对角线
    ax.plot([0, 1], [0, 1], color = 'black', linestyle = '--',linewidth = 3)
    ax.set_xlabel('1-Specificity')
    ax.set_ylabel('Sensitivity')
    ax.grid()
    ax.legend()
    #format(fig)
    st.markdown(":blue[*ROC曲线和AUC数值*]")
    st.plotly_chart(fig,theme="streamlit")#  use_container_width=True
    st.info("单样本实时预测")

    col1, col2,col3 = st.columns(3)

    with col1:
        feature1=st.text_input('sepal length (cm)',1)
        feature2=st.text_input('sepal width (cm)',1)
    with col2:
        feature3=st.text_input('petal length (cm)',1)
        feature4=st.text_input('petal width (cm)',1)


    data_to_pred=pd.DataFrame({'sepal length (cm)':[feature1],'sepal width (cm)':[feature2],'petal length (cm)':[feature3],'petal width (cm)':[feature4]})
    result=clf.predict(data_to_pred)
    with col3:
        st.info("预测结果为：")
        if result==0:
            st.write('**setosa**')
        elif result==1:
            st.write('**versicolor**')
        else:
            st.write('**virginica**')
    st.info("多样本实时预测")    
    uploaded_file = st.file_uploader("Choose a csv file")

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        try:
            dataframe=dataframe[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
            dataframe["predict"]=clf.predict(dataframe)
            dataframe["predict"]=dataframe["predict"].replace({0:'setosa',1:'versicolor',2:'virginica'})
            csv = convert_df(dataframe)
            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='predict_result.csv',
            mime='text/csv')
        except:
            st.write("The file has errors,please check it!") 

def main():
    # 设置初始页面为Home
    session_state = st.session_state
    if 'page' not in session_state:
        session_state['page'] = '数据可视化'

    # 导航栏
    page = st.sidebar.radio('导航栏', ['数据可视化', '模型'])

    if page == '数据可视化':
        page_Visualization()
    elif page == '模型':
        page_model()

if __name__ == '__main__':
    main()

