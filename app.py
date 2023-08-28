#import libraries
from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo
import pickle
import shap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import uuid
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, FactorRange, HoverTool
from bokeh.transform import dodge, transform
from dotenv import load_dotenv
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

#configure flask and connect to MongoDB server
load_dotenv()
app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

#create collections in Mongodb
toddler_collection = mongo.db.toddler_collection
child_collection = mongo.db.child_collection
adolescent_collection = mongo.db.adolescent_collection
adult_collection = mongo.db.adult_collection 


#load models and training data (needed for SHAP values)

with open('toddler_xgb.pkl', 'rb') as file:
    toddlermodel = pickle.load(file)
toddler_X_train = pd.read_csv("toddlerXtrain.csv")


with open('child_xgb.pkl', 'rb') as file:
    childmodel = pickle.load(file)
child_X_train = pd.read_csv("childXtrain.csv")


with open('adult_xgb_new.pkl', 'rb') as file:
    adultmodel = pickle.load(file)

adult_X_train = pd.read_csv("adultXtrain.csv")

#load scalers for pre-processing



#create basic routes for application
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/links')
def links():
    return render_template('links.html')


#create routes for first page of test (demographic info)
@app.route('/toddlertest')
def toddler():
    return render_template('toddlertest.html')

@app.route('/childtest')
def child():
    return render_template('childtest.html')

@app.route('/adolescenttest')
def adolescent():
    return render_template('adolescenttest.html')

@app.route('/adulttest')
def adult():
    return render_template('adulttest.html')


#route for second page of questions
@app.route("/toddlertest/start", methods = ["GET", "POST"])
def toddlerstart():
    #retrieve data  from the first page of questions
   

     Age_Mons = int(request.form["Age_Mons"])
     Sex = int(request.form["Sex"])
     Jaundice = int(request.form["Jaundice"])
     Family_mem_with_ASD = int(request.form["Family_mem_with_ASD"])

     info = [[Age_Mons, Sex, Jaundice, Family_mem_with_ASD]]


     #create unique id and add data to mongodb collection 
     unique_id = str(uuid.uuid4())
     result = {"_id": unique_id, "info": info}
     toddler_collection.insert_one(result)

        

     return redirect(url_for('toddlerquestions', unique_id = unique_id))

   
@app.route("/toddlertest/questions/<unique_id>", methods = ["GET"])
def toddlerquestions(unique_id):
          
      data = toddler_collection.find_one({"_id": unique_id})

      if data is None:
        return "No data found for the provided ID", 404
      
      print(data['info'])
      
      
      Sex = data["info"][0][1]
      gender_text = []

      if Sex == 0:
        gender = "she"
      elif Sex ==1:
        gender = "he"
      else:
        gender = "error"

      gender_text.append(gender)

      his_her = []

      if Sex == 0:
        gender = "her"
      elif Sex ==1:
        gender = "his"
      else:
        gender = "error"

      his_her.append(gender)      


      return render_template('toddlerquestions.html', gender_text = gender_text[0], his_her = his_her[0], unique_id=unique_id)
       
   
@app.route("/toddlertest/questions/predict/<unique_id>", methods = ['POST'])
def toddlerpredict(unique_id):
  
       data = toddler_collection.find_one({"_id": unique_id})

       if data is None:
         return "No data found for the provided ID", 404
       
       #retrieve first page data from mongodb
       info = data["info"]


       #retrieve question data from form 
       if request.method == 'POST': 
          A1_string = request.form["A1"]
          A2_string = request.form["A2"]
          A3_string = request.form["A3"]
          A4_string = request.form["A4"]
          A5_string = request.form["A5"]
          A6_string = request.form["A6"]
          A7_string = request.form["A7"]
          A8_string = request.form["A8"]
          A9_string = request.form["A9"]
          A10_string = request.form["A10"]
           
          #create list of string data 
          input_strings = [[A1_string, A2_string, A3_string, A4_string, A5_string, A6_string, A7_string, A8_string, A9_string, A10_string]]
          
          #convert string data to integers for model 
          A1 = 1
          if A1_string == "Always" or  A1_string =="Usually": 
             A1 = 0
          A2 =1
          if A2_string =="Very easy" or A2_string == "Quite easy":
             A2 =0
          A3 =1
          if A3_string =="Many times a day" or A3_string == "A few times a day":
             A3 =0
          A4 =1
          if A4_string =="Many times a day" or A2_string == "A few times a day":
             A4 =0
          A5 =1
          if A5_string =="Many times a day" or A5_string == "A few times a day":
             A5 =0
          A6 =1
          if A6_string =="Many times a day" or A6_string == "A few times a day":
             A6 =0
          A7 =1
          if A7_string =="Always" or A7_string == "Usually":
             A7 =0
          A8 =1
          if A8_string =="Very typical" or A8_string == "Quite typical":
             A8 =0
          A9 =1
          if A9_string =="Many times a day" or A9_string == "A few times a day":
             A9 =0
          A10 =0
          if A10_string =="Many times a day" or A10_string == "A few times a day":
             A10 =1

          #create list of integer data 
          input_cols = [[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]]
          #add both sets of integer data for prediction
          pred_data = input_cols[0] + info[0]
      

          columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Age_Mons", "Sex", "Jaundice", "Family_mem_with_ASD"]
          prediction_df = pd.DataFrame([pred_data], columns=columns)



          #create prediction from model 
          prediction = toddlermodel.predict(prediction_df)

          #create prediction text for prediction page 
          predicted_texts = []
          pred_result = []
          for pred in prediction:
            if  pred == 0:
             predicted_text = "The screening test has not identified any autistic traits."
             result = 0
            elif  pred == 1:
              predicted_text = "Your child may have autistic traits. Consider seeking a clinical assessment."
              result = 1
            else:
              predicted_text = "Prediction error"
              result = "error"
          
          predicted_texts.append(predicted_text)
          pred_result.append(result)

         # update the mongodb document with new data from form (integers and the input strings)
          toddler_collection.update_one({"_id": unique_id}, {"$set": {"input_cols": input_cols ,"input_strings": input_strings, "pred_result": pred_result}})

          pathway = "/toddlertest/questions/predict"

       
       return render_template("predict.html", prediction_text = predicted_texts[0], unique_id = unique_id, pathway = pathway)

       
#route for explainable AI page 
@app.route("/toddlertest/predict/xai/<unique_id>", methods = ["GET"])
def toddler_plot(unique_id):
       #retrieve data from toddler collection 
      data = toddler_collection.find_one({"_id": unique_id})

      if data is None:
         return "No data found for the provided ID", 404
      
      #retrieve arrays from document 
      info = data["info"]
      input_cols = data["input_cols"]
      input_strings = data["input_strings"]
      pred_result = data["pred_result"]
      
      #create text string for the info questions 
      info_answers = []
      info_answers.append(info[0][0])
      if info[0][1] ==0:
         sex = "Female"
      else:
         sex = "Male"
      info_answers.append(sex)
      if info[0][2] == 0:
         jaundice = "No"
      else: 
         jaundice = "Yes"
      info_answers.append(jaundice)
      if info[0][3] == 0:
         family = "No"
      else:
         family = "Yes"
      info_answers.append(family)

      #create list of prediction data from mongoDB arrays
      pred_data = input_cols[0] + info[0]
      score = sum(input_cols[0])
      prediction_data = [pred_data]
      input_tensor = prediction_data[0]

      print(input_tensor)

      #reshape to use data to create shap values
      reshaped_data = np.array(input_tensor).reshape(1, -1)
       
      #create array from x_train data
      X_train_array = toddler_X_train.values
      background = X_train_array
      
      #create SHAP explainer using model and training data
      explainer = shap.TreeExplainer(toddlermodel, background )
      shap_values = explainer.shap_values(reshaped_data)

      print('shap values:')
      print(shap_values)
      print('shap values shape')
      print(shap_values.shape)

      print(type(shap_values))
      if isinstance(shap_values, np.ndarray):
        print(shap_values.shape)

      print(shap_values[0])

      #function for SHAP value bokeh plot 
      def create_bokeh_plot(shap_values, feature_names, questions, answers):
         #defines data lists
         data={"feature_names": feature_names, "shaps": shap_values, "questions": questions, "answers": answers}

         #sort the features in order of magnitude of shap values
      
         sorted_indices = sorted(range(len(shap_values)), key=lambda i: abs(shap_values[i]), reverse=False)
         sorted_feature_names = FactorRange(*[feature_names[i] for i in sorted_indices])
         
         #function for color of bars (positive direction = red, negative = blue )
         def color_mapper(value):
           return "red" if value > 0 else "blue"
         
         #define colours using function
         data["colors"] = [color_mapper(val) for val in shap_values]
         
         #define data sources
         source = ColumnDataSource(data)
         
         #tools - removed hover and crosshair
         TOOLS = "reset,save"

         # determine figure 
         p = figure(y_range = sorted_feature_names, height = 600, width = 800, x_axis_label = "Feature importance", tools = TOOLS)

         #info to display on hover tool
         hover = HoverTool(tooltips=[("Feature", "@feature_names"), ("Description", "@questions"), ("Your Response", "@answers")])
         p.add_tools(hover)
         #define bars for plot 
         p.hbar( y = dodge("feature_names", 0, range = p.y_range), height = 0.7, left = 0, right = "shaps", source =source, color = "colors")

         p.ygrid.grid_line_color = None
         p.axis.minor_tick_line_color = None
         p.outline_line_color = None
         p.xaxis.axis_label_standoff = 12

         return p
      
      #define data lists for plot
      shap_list = shap_values[0]
      feature_names = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Age", "Sex", "Jaundice", "Family member with ASD"]
      questions = [
         "Does your child look at you when you call their name? ",
         "How easy is it for you to get eye contact with your child?",
         "Does you child point to indicate that they want something?",
         "Does your child point to share interest with you?(e.g. pointing at an interesting sight)",
         "Does your child pretend ? (e.g. care for dolls, talk on a toy phone)",
         "Does your child follow where you're looking?",
         "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)",
         "How would you describe your childs first words?",
         "Does your child use simple gestures (e.g. wave goodbye)",
         "Does your child stare at nothing with no apparent purpose?",
         "Age in months",
         "Sex",
         "Was your child born with jaundice?",
         "Do any family members have ASD?"
         ]
      answers = input_strings[0] + info_answers
      
      #create plot using function and relevant values 
      plot = create_bokeh_plot(shap_list, feature_names, questions, answers)
      script, div = components(plot)

      result_text = []
      if pred_result[0] ==0:
         result = " Not likely to have ASD traits."
      elif pred_result[0] ==1:
         result = " May have ASD traits."
      else:
         result = "prediction error"
      result_text.append(result)

      return render_template("xai.html", script = script, div = div, score = score, result_text = result_text[0] )


#Same routes for Child questionnaire 
@app.route("/childtest/start", methods = ["GET", "POST"])
def childstart():
    if request.method == 'POST': 

     Age = int(request.form["age"])
     Sex = int(request.form["gender"])
     Jaundice = int(request.form["jaundice"])
     

     info = [[Age, Sex, Jaundice]]

     unique_id = str(uuid.uuid4())
     result = {"_id": unique_id, "info": info}
     child_collection.insert_one(result)

    
      
     return redirect(url_for("childquestions", unique_id = unique_id))

@app.route("/childtest/questions/<unique_id>", methods = ["GET"])
def childquestions(unique_id):
          
      data = child_collection.find_one({"_id": unique_id})
      if data is None:
        return "No data found for the provided ID", 404
      
      Sex = data["info"][0][1]

      gender_text = []

      if Sex == 0:
        gender = "she"
      elif Sex ==1:
        gender = "he"
      else:
        gender = "error"

      gender_text.append(gender)

      gender_text_upper = []

      if Sex == 0:
        gender = "She"
      elif Sex ==1:
        gender = "He"
      else:
        gender = "error"

      gender_text_upper.append(gender)

      his_her = []

      if Sex == 0:
        gender = "her"
      elif Sex ==1:
        gender = "his"
      else:
        gender = "error"

      his_her.append(gender)  
      
      
      return render_template('childquestions.html', gender_text = gender_text[0], gender_text_upper = gender_text_upper[0], his_her = his_her[0], unique_id=unique_id)
       

     
@app.route("/childtest/questions/predict/<unique_id>", methods = ["POST"])
def childpredict(unique_id):
       
       data = child_collection.find_one({"_id": unique_id})

       if data is None:
         return "No data found for the provided ID", 404
       
       info = data["info"]

           #retrieve question data from form 
       if request.method == 'POST': 
          A1_string = request.form["A1_Score"]
          A2_string = request.form["A2_Score"]
          A3_string = request.form["A3_Score"]
          A4_string = request.form["A4_Score"]
          A5_string = request.form["A5_Score"]
          A6_string = request.form["A6_Score"]
          A7_string = request.form["A7_Score"]
          A8_string = request.form["A8_Score"]
          A9_string = request.form["A9_Score"]
          A10_string = request.form["A10_Score"]
           
          #create list of string data 
          input_strings = [[A1_string, A2_string, A3_string, A4_string, A5_string, A6_string, A7_string, A8_string, A9_string, A10_string]]
          
          #convert string data to integers for model 
          
          def string_to_var_1(string):
             if string == "Definitely agree" or  string =="Slightly agree": 
               variable = 1
             elif string == "Slightly disagree" or string == "Definitely disagree":
               variable = 0
             else: 
                return "Error", 404
             return variable
          
          def string_to_var_0(string):
             if string == "Definitely agree" or  string =="Slightly agree": 
               variable = 0
             elif string == "Slightly disagree" or string == "Definitely disagree":
               variable = 1
             else: 
                return "Error", 404
             return variable
          
          A1 = string_to_var_1(A1_string)
          A2 = string_to_var_0(A2_string)
          A3 = string_to_var_0(A3_string)
          A4 = string_to_var_0(A4_string)
          A5 = string_to_var_1(A5_string)
          A6 = string_to_var_0(A6_string)
          A7 = string_to_var_1(A7_string)
          A8 = string_to_var_0(A8_string)
          A9 = string_to_var_0(A9_string)
          A10 = string_to_var_1(A10_string)

          input_cols = [[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]]

          pred_data = input_cols[0] + info[0]

          columns = ["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score", "age", "gender", "jaundice"]
          prediction_df = pd.DataFrame([pred_data], columns=columns)


          prediction = childmodel.predict(prediction_df)

          prediction_data = [pred_data]

       
         
          predicted_texts = []
          pred_result = []
          for pred in prediction:
           if  pred ==0 :
             predicted_text = "The screening test has not identifed any autistic traits."
             result = 0
           elif  pred ==1:
              predicted_text = "Your child may have autistic traits. Consider seeking a clinical assessment."
              result = 1
           else:
              predicted_text = "Prediction error"
              result = "error"
          
          predicted_texts.append(predicted_text)
          pred_result.append(result)

          child_collection.update_one({"_id": unique_id}, {"$set": {"input_cols": input_cols, "input_strings": input_strings, "pred_result": pred_result}})

          pathway = "/childtest/questions/predict"

       
       return render_template("predict.html", prediction_text = predicted_texts[0], unique_id = unique_id, prediction_data = prediction_data, pathway = pathway)


#route for explainable AI page 
@app.route("/childtest/predict/xai/<unique_id>", methods = ["GET"])
def child_plot(unique_id):
       #retrieve data from toddler collection 
      data = child_collection.find_one({"_id": unique_id})

      if data is None:
         return "No data found for the provided ID", 404
      
      #retrieve arrays from document 
      info = data["info"]
      input_cols = data["input_cols"]
      input_strings = data["input_strings"]
      pred_result = data["pred_result"]
      
      #create text string for the info questions 
      info_answers = []
      info_answers.append(info[0][0])
      if info[0][1] ==0:
         sex = "Female"
      else:
         sex = "Male"
      info_answers.append(sex)
      if info[0][2] == 0:
         jaundice = "No"
      else: 
         jaundice = "Yes"
      info_answers.append(jaundice)

      #create list of prediction data from mongoDB arrays
      pred_data = input_cols[0] + info[0]
      score = sum(input_cols[0])
      prediction_data = [pred_data]
      input_tensor = prediction_data[0]

      #reshape to use data to create shap values
      reshaped_data = np.array(input_tensor).reshape(1, -1)
       
      #create array from x_train data
      X_train_array = child_X_train.values
      background = X_train_array
      #create SHAP explainer using model and training data
      explainer = shap.TreeExplainer(adultmodel, background )
      shap_values = explainer.shap_values(reshaped_data)


      #function for SHAP value bokeh plot 
      def create_bokeh_plot(shap_values, feature_names, questions, answers):
         #defines data lists
         data={"feature_names": feature_names, "shaps": shap_values, "questions": questions, "answers": answers}

         #sort the features in order of magnitude of shap values
         sorted_indices = sorted(range(len(shap_values)), key=lambda i: abs(shap_values[i]), reverse=False)
         sorted_feature_names = FactorRange(*[feature_names[i] for i in sorted_indices])
         
         #function for color of bars (positive direction = red, negative = blue )
         def color_mapper(value):
           return "red" if value > 0 else "blue"
         
         #define colours using function
         data["colors"] = [color_mapper(val) for val in shap_values]
         
         #define data sources
         source = ColumnDataSource(data)
         
         #tools - removed hover and crosshair
         TOOLS = "reset,save"

         # determine figure 
         p = figure(y_range = sorted_feature_names, height = 600, width = 800, x_axis_label = "Feature importance", tools = TOOLS)

         #info to display on hover tool
         hover = HoverTool(tooltips=[("Feature", "@feature_names"), ("Description", "@questions"), ("Your Response", "@answers")])
         p.add_tools(hover)
         #define bars for plot 
         p.hbar( y = dodge("feature_names", 0, range = p.y_range), height = 0.7, left = 0, right = "shaps", source =source, color = "colors")

         p.ygrid.grid_line_color = None
         p.axis.minor_tick_line_color = None
         p.outline_line_color = None
         p.xaxis.axis_label_standoff = 12

         return p
      
      #define data lists for plot
      shap_list = shap_values[0]
      feature_names = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Age", "Sex", "Jaundice"]
      questions = [
         "He/she often notices small sounds when others do not.",
         "He/she usually concentrates more on the whole picture, rather than small details.",
         "In a social group, he/she can easily keep track of several different people's conversations",
         "He/she finds it easy to go back and forth between different activities.",
         "He/she doesn't know how to keep a conversation going with his/her peers.",
         "He/she is good at social chit-chat",
         "When he/she is read a story, he/she finds it difficult to work out the character's intentions",
         "When he/she was in pre-school, they used to enjoy playing games involving pretending with other children.",
         "He/she finds it easy to work out what someone is thinking or feeling just by looking at their face",
         "He/she finds it hard to make new friends.",
         "Age in years",
         "Gender",
         "Was your child born with jaundice?"
         ]
      answers = input_strings[0] + info_answers
      
      #create plot using function and relevant values 
      plot = create_bokeh_plot(shap_list, feature_names, questions, answers)
      script, div = components(plot)

      result_text = []
      if pred_result[0] ==0:
         result = " Not likely to have ASD traits."
      elif pred_result[0] ==1:
         result = " May have ASD traits."
      else:
         result = "prediction error"
      result_text.append(result)

      return render_template("xai.html", script = script, div = div, score = score, result_text= result_text[0] )





@app.route("/adolescenttest/start", methods = ["GET", "POST"])
def adolescentstart():
    if request.method == 'POST': 

     Age = int(request.form["age"])
     Sex = int(request.form["gender"])
     Jaundice = int(request.form["jaundice"])
     

     info = [[Age, Sex, Jaundice]]

     unique_id = str(uuid.uuid4())
     result = {"_id": unique_id, "info": info}
     adolescent_collection.insert_one(result)

     
        

     return redirect(url_for("adolescentquestions", unique_id = unique_id))

@app.route("/adolescenttest/questions/<unique_id>", methods = ["GET"])
def adolescentquestions(unique_id):
          
      data = adolescent_collection.find_one({"_id": unique_id})
      if data is None:
        return "No data found for the provided ID", 404
      
      Sex = data["info"][0][1]
     
      gender_text = []

      if Sex == 0:
        gender = "she"
      elif Sex ==1:
        gender = "he"
      else:
        gender = "error"

      gender_text.append(gender)

      his_her = []

      if Sex == 0:
        gender = "her"
      elif Sex ==1:
        gender = "his"
      else:
        gender = "error"

      his_her.append(gender)

      gender_text_upper = []

      if Sex == 0:
        gender = "She"
      elif Sex ==1:
        gender = "He"
      else:
        gender = "error"

      gender_text_upper.append(gender)

      
      return render_template('adolescentquestions.html',  gender_text = gender_text[0], gender_text_upper = gender_text_upper[0], his_her = his_her[0],unique_id=unique_id)
       


@app.route("/adolescenttest/questions/predict/<unique_id>", methods = ["POST"])
def adolescentpredict(unique_id):
       
       data = adolescent_collection.find_one({"_id": unique_id})

       if data is None:
         return "No data found for the provided ID", 404
       
       info = data["info"]


      #retrieve question data from form 
       if request.method == 'POST': 
          A1_string = request.form["A1_Score"]
          A2_string = request.form["A2_Score"]
          A3_string = request.form["A3_Score"]
          A4_string = request.form["A4_Score"]
          A5_string = request.form["A5_Score"]
          A6_string = request.form["A6_Score"]
          A7_string = request.form["A7_Score"]
          A8_string = request.form["A8_Score"]
          A9_string = request.form["A9_Score"]
          A10_string = request.form["A10_Score"]
           
          #create list of string data 
          input_strings = [[A1_string, A2_string, A3_string, A4_string, A5_string, A6_string, A7_string, A8_string, A9_string, A10_string]]
          
          #convert string data to integers for model 
          
          def string_to_var_1(string):
             if string == "Definitely agree" or  string =="Slightly agree": 
               variable = 1
             elif string == "Slightly disagree" or string == "Definitely disagree":
               variable = 0
             else: 
                return "Error", 404
             return variable
          
          def string_to_var_0(string):
             if string == "Definitely agree" or  string =="Slightly agree": 
               variable = 0
             elif string == "Slightly disagree" or string == "Definitely disagree":
               variable = 1
             else: 
                return "Error", 404
             return variable
          
          A1 = string_to_var_1(A1_string)
          A2 = string_to_var_0(A2_string)
          A3 = string_to_var_0(A3_string)
          A4 = string_to_var_0(A4_string)
          A5 = string_to_var_1(A5_string)
          A6 = string_to_var_0(A6_string)
          A7 = string_to_var_1(A7_string)
          A8 = string_to_var_0(A8_string)
          A9 = string_to_var_0(A9_string)
          A10 = string_to_var_1(A10_string)

          input_cols = [[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]]

          pred_data = input_cols[0] + info[0]

          columns = ["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score", "age", "gender", "jaundice"]
          prediction_df = pd.DataFrame([pred_data], columns=columns)

          prediction_data = [pred_data]
          print("prediction data:") 
          print(prediction_data)

          print("pred data:")
          print(pred_data)


          prediction = childmodel.predict(prediction_df)

          print(prediction)

          predicted_texts = []
          pred_result = []
          for pred in prediction:
           if  pred==0:
             predicted_text = "The screening test has not identified any autistic traits."
             result = 0
           elif  pred==1:
              predicted_text = "Your child may have autistic traits. Consider seeking a clinical assessment."
              result = 1
           else:
              predicted_text = "Prediction error"
          
          predicted_texts.append(predicted_text)
          pred_result.append(result)

          adolescent_collection.update_one({"_id": unique_id}, {"$set": {"input_cols": input_cols, "input_strings": input_strings, "pred_result": pred_result}})
          pathway = "/adolescenttest/questions/predict"

       
       return render_template("predict.html", prediction_text = predicted_texts[0], unique_id = unique_id, prediction_data = prediction_data, pathway = pathway)

#route for explainable AI page 
@app.route("/adolescenttest/predict/xai/<unique_id>", methods = ["GET"])
def adolescent_plot(unique_id):
       #retrieve data from toddler collection 
      data = adolescent_collection.find_one({"_id": unique_id})

      if data is None:
         return "No data found for the provided ID", 404
      
      #retrieve arrays from document 
      info = data["info"]
      input_cols = data["input_cols"]
      input_strings = data["input_strings"]
      pred_result = data["pred_result"]
      
      #create text string for the info questions 
      info_answers = []
      info_answers.append(info[0][0])
      if info[0][1] ==0:
         sex = "Female"
      else:
         sex = "Male"
      info_answers.append(sex)
      if info[0][2] == 0:
         jaundice = "No"
      else: 
         jaundice = "Yes"
      info_answers.append(jaundice)

      #create list of prediction data from mongoDB arrays
      pred_data = input_cols[0] + info[0]
      score = sum(input_cols[0])
      prediction_data = [pred_data]
      input_tensor = prediction_data[0]

      #reshape to use data to create shap values
      reshaped_data = np.array(input_tensor).reshape(1, -1)
       
      #create array from x_train data
      X_train_array = child_X_train.values
      background = X_train_array
      #create SHAP explainer using model and training data
      explainer = shap.TreeExplainer(childmodel, background )
      shap_values = explainer.shap_values(reshaped_data)
      

      #function for SHAP value bokeh plot 
      def create_bokeh_plot(shap_values, feature_names, questions, answers):
         #defines data lists
         data={"feature_names": feature_names, "shaps": shap_values, "questions": questions, "answers": answers}

         #sort the features in order of magnitude of shap values
         sorted_indices = sorted(range(len(shap_values)), key=lambda i: abs(shap_values[i]), reverse=False)
         sorted_feature_names = FactorRange(*[feature_names[i] for i in sorted_indices])
         
         #function for color of bars (positive direction = red, negative = blue )
         def color_mapper(value):
           return "red" if value > 0 else "blue"
         
         #define colours using function
         data["colors"] = [color_mapper(val) for val in shap_values]
         
         #define data sources
         source = ColumnDataSource(data)
         
         #tools - removed hover and crosshair
         TOOLS = "reset,save"

         # determine figure 
         p = figure(y_range = sorted_feature_names, height = 600, width = 800, x_axis_label = "Feature importance", tools = TOOLS)

         #info to display on hover tool
         hover = HoverTool(tooltips=[("Feature", "@feature_names"), ("Description", "@questions"), ("Your Response", "@answers")])
         p.add_tools(hover)
         #define bars for plot 
         p.hbar( y = dodge("feature_names", 0, range = p.y_range), height = 0.7, left = 0, right = "shaps", source =source, color = "colors")

         p.ygrid.grid_line_color = None
         p.axis.minor_tick_line_color = None
         p.outline_line_color = None
         p.xaxis.axis_label_standoff = 12

         return p
      
      #define data lists for plot
      shap_list = shap_values[0]
      feature_names = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Age", "Sex", "Jaundice"]
      questions = [
         "He/she often notices patterns in things all the time",
         "He/she usually concentrates more on the whole picture, rather than small details",
         "In a social group, he/she can easily keep track of several different people's conversations",
         "If there is an interruption, he/she can switch back to what they were doing very quickly",
         "He/she frequently finds that they doesn't know how to keep a conversation going with their peers.",
         "He/she is good at social chit-chat",
         "He/she finds it difficult to imagien what it would be like to be someone else",
         "When he/she was younger, they used to enjoy playing games involving pretending with other children.",
         "He/she finds social interactions easy",
         "He/she finds it hard to make new friends",
         "Age in years",
         "Gender",
         "Was your child born with jaundice?"
         ]
      answers = input_strings[0] + info_answers
      
      #create plot using function and relevant values 
      plot = create_bokeh_plot(shap_list, feature_names, questions, answers)
      script, div = components(plot)
      
      result_text = []
      if pred_result[0] ==0:
         result = " Not likely to have ASD traits."
      elif pred_result[0] ==1:
         result = " May have ASD traits."
      else:
         result = "prediction error"
      result_text.append(result)

      return render_template("xai.html", script = script, div = div, score = score, result_text = result_text[0] )




@app.route("/adulttest/start", methods = ["GET", "POST"])
def adultstart():
    if request.method == 'POST': 

     Age = int(request.form["age"])
     Sex = int(request.form["gender"])
     Jaundice = int(request.form["jaundice"])


     info = [[Age, Sex, Jaundice]]

     unique_id = str(uuid.uuid4())
     result = {"_id": unique_id, "info": info}
     adult_collection.insert_one(result)

        

     return redirect(url_for("adultquestions",  unique_id = unique_id))

@app.route("/adulttest/questions/<unique_id>", methods = ["GET"])
def adultquestions(unique_id):
          
      data = adult_collection.find_one({"_id": unique_id})
      if data is None:
        return "No data found for the provided ID", 404
      
      return render_template('adultquestions.html', unique_id=unique_id)

@app.route("/adult/questions/predict/<unique_id>", methods = ["POST"])
def adultpredict(unique_id):
       
       data = adult_collection.find_one({"_id": unique_id})

       if data is None:
         return "No data found for the provided ID", 404
       
       info = data["info"]

       #retrieve question data from form 
       if request.method == 'POST': 
          A1_string = request.form["A1_Score"]
          A2_string = request.form["A2_Score"]
          A3_string = request.form["A3_Score"]
          A4_string = request.form["A4_Score"]
          A5_string = request.form["A5_Score"]
          A6_string = request.form["A6_Score"]
          A7_string = request.form["A7_Score"]
          A8_string = request.form["A8_Score"]
          A9_string = request.form["A9_Score"]
          A10_string = request.form["A10_Score"]
           
          #create list of string data 
          input_strings = [[A1_string, A2_string, A3_string, A4_string, A5_string, A6_string, A7_string, A8_string, A9_string, A10_string]]
          
          #convert string data to integers for model 
          
          def string_to_var_1(string):
             if string == "Definitely agree" or  string =="Slightly agree": 
               variable = 1
             elif string == "Slightly disagree" or string == "Definitely disagree":
               variable = 0
             else: 
                return "Error", 404
             return variable
          
          def string_to_var_0(string):
             if string == "Definitely agree" or  string =="Slightly agree": 
               variable = 0
             elif string == "Slightly disagree" or string == "Definitely disagree":
               variable = 1
             else: 
                return "Error", 404
             return variable
          
          A1 = string_to_var_1(A1_string)
          A2 = string_to_var_0(A2_string)
          A3 = string_to_var_0(A3_string)
          A4 = string_to_var_0(A4_string)
          A5 = string_to_var_0(A5_string)
          A6 = string_to_var_0(A6_string)
          A7 = string_to_var_1(A7_string)
          A8 = string_to_var_1(A8_string)
          A9 = string_to_var_0(A9_string)
          A10 = string_to_var_1(A10_string)

          input_cols = [[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]]

          pred_data = input_cols[0] + info [0]

          prediction_data = [pred_data]


          print("prediction data:") 
          print(prediction_data)

          print("pred data:")
          print(pred_data)

          columns = ["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score", "age", "gender", "jaundice"]
          prediction_df = pd.DataFrame([pred_data], columns=columns)


          prediction = adultmodel.predict(prediction_df)

          print(prediction)

          predicted_texts = []
          pred_result = []
          for pred in prediction:
           if  pred ==0:
             predicted_text = "The screening test has not identifed any autistic traits."
             result = 0
           elif  pred==1:
              predicted_text = "You may have autistic traits. Consider seeking a clinical assessment."
              result = 1
           else:
              predicted_text = "Prediction error"
          
          predicted_texts.append(predicted_text)
          pred_result.append(result)

          adult_collection.update_one({"_id": unique_id}, {"$set": {"input_cols": input_cols, "input_strings": input_strings, "pred_result":pred_result}})

          pathway = "/adulttest/questions/predict"

       
       return render_template("predict.html", prediction_text = predicted_texts[0], unique_id = unique_id, prediction_data = prediction_data, pathway = pathway)

#route for explainable AI page 
@app.route("/adulttest/predict/xai/<unique_id>", methods = ["GET"])
def adult_plot(unique_id):
       #retrieve data from toddler collection 
      data = adult_collection.find_one({"_id": unique_id})

      if data is None:
         return "No data found for the provided ID", 404
      
      #retrieve arrays from document 
      info = data["info"]
      input_cols = data["input_cols"]
      input_strings = data["input_strings"]
      pred_result = data["pred_result"]
      
      #create text string for the info questions 
      info_answers = []
      info_answers.append(info[0][0])
      if info[0][1] ==0:
         sex = "Female"
      else:
         sex = "Male"
      info_answers.append(sex)
      if info[0][2] == 0:
         jaundice = "No"
      else: 
         jaundice = "Yes"
      info_answers.append(jaundice)

      #create list of prediction data from mongoDB arrays
      pred_data = input_cols[0] + info[0]
      score = sum(input_cols[0])
      prediction_data = [pred_data]
      input_tensor = prediction_data[0]

      #reshape to use data to create shap values
      reshaped_data = np.array(input_tensor).reshape(1, -1)
       
      #create array from x_train data
      X_train_array = adult_X_train.values
      background = X_train_array
      print("background")
      print(background)
      
      
      #create SHAP explainer using model and training data
      explainer = shap.TreeExplainer(adultmodel, background )
      shap_values = explainer.shap_values(reshaped_data)

      #function for SHAP value bokeh plot 
      def create_bokeh_plot(shap_values, feature_names, questions, answers):
         #defines data lists
         data={"feature_names": feature_names, "shaps": shap_values, "questions": questions, "answers": answers}

         #sort the features in order of magnitude of shap values
         sorted_indices = sorted(range(len(shap_values)), key=lambda i: abs(shap_values[i]), reverse=False)
         sorted_feature_names = FactorRange(*[feature_names[i] for i in sorted_indices])
         
         #function for color of bars (positive direction = red, negative = blue )
         def color_mapper(value):
           return "red" if value > 0 else "blue"
         
         #define colours using function
         data["colors"] = [color_mapper(val) for val in shap_values]
         
         #define data sources
         source = ColumnDataSource(data)
         
         #tools - removed hover and crosshair
         TOOLS = "reset,save"

         # determine figure 
         p = figure(y_range = sorted_feature_names, height = 600, width = 800, x_axis_label = "Feature importance", tools = TOOLS)

         #info to display on hover tool
         hover = HoverTool(tooltips=[
            ("Feature", "@feature_names"), 
            ("Description", "@questions"), 
            ("Your Response", "@answers")
            
            ])
         
         
         p.add_tools(hover)
         #define bars for plot 
         p.hbar( y = dodge("feature_names", 0, range = p.y_range), height = 0.7, left = 0, right = "shaps", source =source, color = "colors")

         p.ygrid.grid_line_color = None
         p.axis.minor_tick_line_color = None
         p.outline_line_color = None
         p.xaxis.axis_label_standoff = 12

         return p
      
      #define data lists for plot
      shap_list = shap_values[0]
      feature_names = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Age", "Sex", "Jaundice"]
      questions = [
         "I often notice small sounds when others do not",
         "I usually concentrate more on the whole picture, rather than small details",
         "I find it easy to do more than one thing at once",
         "If there is an interruption, I can switch back to what I was doing very quickly",
         "I find it easy to 'read between the lines' when someone is talking to me",
         "I know how to tell if someone listening to me is getting bored",
         "When I'm reading a story I find it difficult to work out the characters' intentions",
         "I like to collect information about categories of things (e.g. types of car, types of bird, types of train, types of plant etc.)",
         "I find it easy to work out what someone is thinking or feeling just by looking at their face",
         "I find it difficult to work out people's intentions",
         "Age in years",
         "Gender",
         "Were you born with jaundice?"
         ]
      answers = input_strings[0] + info_answers
      
      #create plot using function and relevant values 
      plot = create_bokeh_plot(shap_list, feature_names, questions, answers)
      script, div = components(plot)

      result_text = []
      if pred_result[0] ==0:
         result = " Not likely to have ASD traits."
      elif pred_result[0] ==1:
         result = " May have ASD traits."
      else:
         result = "prediction error"
      result_text.append(result)

      return render_template("xai.html", script = script, div = div, score = score, result_text = result_text[0])


if __name__ == '__main__':
    app.run(debug=True)