from text_sim import get_sim_tfidf
from pandas import read_csv
import concurrent.futures
import numpy as np

df = read_csv('processedDataset.csv',names=['txt','label'])
df = df.dropna()
new_df = []
for i in df['txt']:
    new_df.append(i)
list_of_simi=[]
def compare_all_dataset (x):
    for i in range(0,len(new_df),200):
        print (i)
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
            list_of_simi.append(executor.submit(get_sim_tfidf, new_df.pop(),x).result())
        
    #print (list_of_simi)        
    return new_df[np.argmax(list_of_simi)] 
#x = 'ستوديوه رزاز صحراء مرزوك آثار رباط بيضاء نتهى مخرج مغرب سهيل صوير مشاهد سينمائ جديد سليط جاسوس إسبان دومينغ اديا تاسع مغرب فيلم ختار مخرج خليف عباس حياء متنكر تاجر سلال رسول فيما جاسوس حساب إسباني مخرج فيلم سهيل صريح هسبريس فيلم سينمائ مرحل توضيب خارج مغرب مبرز فيلم جاسوس إسبان دومينغ اديا مناطق عالم إسلام جاهز فيلم سينمائ ممثل مختلف ختار بطول ممثل سينمائ إيطال ارول ريشنتين قيام إنجليز هستر ستانهوب اشتهر زنوبي ارتبط علاق عاطف إضاف سينمائ معروف ختيار مخرج مغرب عباس صريح أنباء فرنس حداث مشوق ستحق أضواء مشير فيلم كثير مفاجآت امرأ طموح شجاع مثقف مذهل مستكشف إعجاب مقرب لحاق رحيل عباس مدين مسلم جميع مدين شييد مثال عروس شمال علوم ساعد إخفاء راجع معرف مغارب مسلم'
#compare_all_dataset(x)