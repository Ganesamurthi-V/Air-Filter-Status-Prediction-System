from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Dummy model placeholders — replace with your loaded models
model1 = "tabpfn_airfilter_model.pkl" 
model2 = "voting_classifier_model.pkl"

def fetch_data_from_api(device_id, from_ts, to_ts):
    url = "YOUR_API_URL"
    payload = {"device_Id": device_id, "fromEpoch": from_ts, "toEpoch": to_ts}

    try:
        response = requests.post(url, json=payload, verify=False)
        data = response.json()
        if data.get("status") == "success" and data.get("data"):
            return pd.DataFrame(data["data"])
        logging.warning("API returned no data: %s", data.get("message", "Unknown error"))
    except Exception as e:
        logging.error("Error fetching API data: %s", e)
        traceback.print_exc()
    return pd.DataFrame()  # return empty DataFrame on failure


@app.route('/')
def index():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    final_string1 = ''
    final_string2 = ''

    try:
        device_id = request.form["device_id"]
        dt_str = request.form["datetime"]
        duration_minutes = int(request.form["duration"])

        # Convert timestamp
        date_part, time_part = dt_str.split("T")
        time_part += ":0"
        dt_values = list(map(int, date_part.split("-")))
        time_values = list(map(int, time_part.split(":")))

        from_ts = int(datetime(*dt_values, *time_values).timestamp())
        to_ts = from_ts + duration_minutes * 60

        roomsize_dev_dict = {
            #add your room id and room size 
        }

        dev_ses_df = fetch_data_from_api(device_id, from_ts, to_ts)
        # dev_ses_df.to_csv("test_df_app.csv")

        if dev_ses_df.empty:
            final_string1 = "No data returned from API. Try again later or adjust the time range."
            return render_template('form.html', prediction=final_string1)

        # Data Cleaning
        dev_ses_df['temp_diff'] = dev_ses_df['start_temp'] - dev_ses_df['end_temp']
        dev_ses_df = dev_ses_df[dev_ses_df['temp_diff'] > 0]
        dev_ses_df = dev_ses_df[dev_ses_df['duration'] > 300]

        if len(dev_ses_df) < 1:
            final_string1 = "Not enough valid sessions found. Try increasing duration."
            return render_template('form.html', prediction=final_string1)

        # Predict for each session
        dev_predictions1 = []
        dev_predictions2 = []

        for _, row in dev_ses_df.iterrows():
            try:
                # ... (keep all your transformation code here as-is)
                if len(dev_ses_df)>1:
                    
                    #print("********************",len(dev_ses_df),"****************************")
                    url_dev_per="YOUR_API_URL"
                    for _,row in dev_ses_df.iterrows():
                        aux_list=[]
                        row_df = row.to_frame().T.copy()
                        print(row_df)
                        aux_list.append({'current':np.nan, 'power':np.nan, 
                                                'voltage':np.nan, 'temp':row_df['start_temp'].iloc[0]*10.0,
                                                'rly_sts':np.nan, 'device_Id':row_df['device_Id'].iloc[0],  
                                                'timeStamp':row_df['relay_start_time'].iloc[0]})
                        aux_list.append({'current':np.nan, 'power':np.nan, 
                                                'voltage':np.nan, 'temp':row_df['end_temp'].iloc[0]*10.0,
                                                'rly_sts':np.nan, 'device_Id':row_df['device_Id'].iloc[0],  
                                                'timeStamp':row_df['relay_start_time'].iloc[0]+row_df['duration'].iloc[0]})
                        aux_periodic_df=pd.DataFrame(aux_list)
                        row_per_list=[]
                        
                        row_per_body={
                            "device_Id":device_id,
                            "fromEpoch":row_df['relay_start_time'].iloc[0],
                            "toEpoch":row_df['relay_start_time'].iloc[0]+row_df['duration'].iloc[0]
                        }
                        row_per_resp=requests.post(url_dev_per, json=row_per_body, verify=False).json()
                        if row_per_resp['status']=="success":
                            row_per_list.extend(row_per_resp['data'])
                        else:
                            print("Check")
                        #print(row_per_list)
                        row_per_df=pd.DataFrame(row_per_list)
                        row_per_df['power_cal']=row_per_df['current']*row_per_df['voltage']/1000000
                        row_per_df['power']=row_per_df[['power', 'power_cal']].min(axis=1)
                        row_per_df.drop(columns=['power_cal'],inplace=True)
                        row_per_df['current']=row_per_df['power']*1000/row_per_df['voltage']
                        print(row_per_df)
                        row_per_df.to_csv("test_app.csv")
                        row_per_df=row_per_df[['current', 'power', 'voltage', 'temp', 'rly_sts', 'device_Id', 'timeStamp']]
                        row_per_df=pd.concat([row_per_df, aux_periodic_df])
                        row_per_df.drop_duplicates(subset=['timeStamp'], keep='last', inplace=True)
                        row_per_df=row_per_df.sort_values(by='timeStamp')
                        #row_per_df.sort_values(by='timeStamp', inplace=True)
                        #ses_row_df=pd.DataFrame([row])
                        if len(row_per_df)==0:
                            continue
                        else:
                            # Ensure it's a Series
                            assert isinstance(row_df['device_Id'], pd.Series)
                            assert isinstance(row_per_df['device_Id'], pd.Series)

                            # Ensure no Series or list inside cells
                            assert not row_df['device_Id'].apply(lambda x: isinstance(x, pd.Series)).any()
                            assert not row_per_df['device_Id'].apply(lambda x: isinstance(x, pd.Series)).any()

                            row_per_ses_df=row_df.merge(row_per_df, how='left', on=['device_Id'])
                            row_per_ses_df=row_per_ses_df.sort_values(by='timeStamp')
                            row_power_df=row_per_ses_df[[ 'device_Id','relay_start_time','power', 'timeStamp']]
                            row_temp_frame=row_per_ses_df[['relay_start_time','start_temp','end_temp','temp','timeStamp', 'duration']]
                            row_balance_frame=row_per_ses_df[['relay_start_time','duration','total_energy']]
                            agg_power_df = row_power_df.groupby([ 'device_Id','relay_start_time']).agg(mean_power=('power', 'mean')).reset_index()
                            row_temp_frame['temp_slope_diff']=row_temp_frame.groupby('relay_start_time')['temp'].diff()
                            row_temp_frame['time_stamp_diff']=row_temp_frame.groupby('relay_start_time')['timeStamp'].diff()                
                            row_temp_frame['temp_slope_diff']=(row_temp_frame['temp_slope_diff']*row_temp_frame['time_stamp_diff'])/row_temp_frame['duration']
                            row_temp_frame['temp_slope']=row_temp_frame['temp_slope_diff']/row_temp_frame['time_stamp_diff']
                            row_temp_frame.drop(columns=['temp_slope_diff', 'time_stamp_diff'], inplace=True)
                            agg_temp_df = row_temp_frame.groupby([ 'relay_start_time']).agg(start_temp=('start_temp', 'first'),
                                end_temp=('end_temp', 'first'), min_temp=('temp', 'min'), max_temp=('temp', 'max'),
                                mean_temp=('temp', 'mean'), min_temp_slope=('temp_slope', 'min'), max_temp_slope=('temp_slope', 'max'),
                                mean_temp_slope=('temp_slope', 'mean'), std_temp_slope=('temp_slope', 'std')).reset_index()
                            agg_balance_frame=row_balance_frame.groupby(['relay_start_time']).agg(duration=('duration','mean'),
                                                                                                total_energy=('total_energy','mean')).reset_index()
                            row_full_df=pd.concat([agg_power_df,agg_temp_df, agg_balance_frame], axis=1)
                            row_full_df['direct_temp_slope']=(row_full_df['start_temp']-row_full_df['end_temp'])/row_full_df['duration']

                            # Final preprocessing
                            row_full_df['room_size'] = row_full_df['device_Id'].map(roomsize_dev_dict)
                            row_full_df['energy/cubm'] = row_full_df['total_energy'] / row_full_df['room_size']
                            row_full_df['avg_pwr/cubm'] = row_full_df['energy/cubm'] / row_full_df['duration']
                            row_full_df['mean_pwr/cubm'] = row_full_df['mean_power'] / row_full_df['room_size']
                            row_full_df.drop(columns=['device_Id', 'relay_start_time'], inplace=True)

                            for col in row_full_df.columns:
                                row_full_df[col] = pd.to_numeric(row_full_df[col], errors='coerce')

                            prediction1 = model1.predict(row_full_df)
                            prediction2 = model2.predict(row_full_df)
                            dev_predictions1.extend(prediction1)
                            dev_predictions2.extend(prediction2)
            except Exception as row_err:
                logging.error("Row processing error: %s", row_err)
                continue

        if not dev_predictions1 or not dev_predictions2:
            final_string1 = "Model could not make predictions. Check logs for details."
        else:
            # Summary
            dirty_prob1 = dev_predictions1.count(1.0) / len(dev_predictions1)
            clean_prob1 = 1 - dirty_prob1
            dirty_prob2 = dev_predictions2.count(1.0) / len(dev_predictions2)
            clean_prob2 = 1 - dirty_prob2

            final_string1 = f"Model 1: Air filter is {'dirty' if dirty_prob1 > 0.2 else 'clean'} with {dirty_prob1*100:.2f}%"
            final_string2 = f"Model 2: Air filter is {'dirty' if dirty_prob2 > 0.2 else 'clean'} with {dirty_prob2*100:.2f}%"

    except Exception as e:
        logging.error("Prediction error: %s", e)
        final_string1 = f"Error occurred: {str(e)}"
        traceback.print_exc()

    return render_template('form.html', prediction=final_string1 + "<br>" + final_string2)


if __name__ == '__main__':
    app.run(debug=True)

