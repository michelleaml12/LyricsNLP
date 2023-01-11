###
# This module is intended to select data after all of the gathering and processing is complete.
###



import numpy as np
import pandas as pd
import sqlite3
dbname = 'Lyrics_DB.sqlite'
my_conn = sqlite3.connect(dbname)
cur = my_conn.cursor()

# If in this list, selection is made from the ProcessedSongData table, other data is in the SongData table
processed_types = ["Sentiment", "TokenCount"]

# If in this list, the data averages should be made by equality as there is no ordering or notion of addition
discrete_types = ["MainGenre", "MajorKey"]

# If in this list, they represent a variable suitable to go on the x-axis of a graph
all_x_types = ["ReleaseYear",
"MainGenre",
"PopularityOnQuery",
"Acousticness",
"Danceability",
"Duration",
"Energy",
"MajorKey",
"Speechiness",
"Tempo",
"Valence",
"Sentiment",
"TokenCount"]

# If in this list, they represent a variable suitable to go on the y-axis of a graph
all_y_types = ["PopularityOnQuery",
"Acousticness",
"Danceability",
"Duration",
"Energy",
"ReleaseYear",
"Speechiness",
"Tempo",
"Valence",
"Sentiment",
"TokenCount"]


#Produces a fully qualified column name (in the matching table)
def col_expand(name):
  if name in processed_types:
    return "ProcessedSongData."+name
  return "SongData."+name


# Returns:
# new_arr1 : The unique elements of arr1, sorted alphabetically
# new_arr2 : For any index n, new_arr2[n] is the average of the y-value of all data points whose x-value is new_arr1[n]
def collapse_by_equality(arr1, arr2):
  sorted_indices = np.argsort(arr1)
  arr1 = np.array(arr1)[sorted_indices].tolist()
  arr2 = np.array(arr2, dtype="float64")[sorted_indices].tolist()
  new_arr1 = []
  new_arr2 = []
  while(len(arr1)>0):
    x1 = arr1[0]
    n = 0
    while(arr1[n]==x1 and n+1 < len(arr1)):
      n+=1
    new_arr1.append(x1)
    new_arr2.append(np.round(np.mean(arr2[0:n+1]),2))
    if len(arr1)==n:
      break
    arr1 = arr1[n+1:]
    arr2 = arr2[n+1:]
  return new_arr1, new_arr2

# Returns:
# new_arr1 : The unique elements of arr1, sorted alphabetically, where elements within 0.01 of the most recently added to arr1 are not considered unique
# new_arr2 : For any index n, new_arr2[n] is the average of the y-value of all data points whose x-value meets (new_arr1[n] <= x <= new_arr1[n] + 0.01)
def collapse_within_range(arr1, arr2):
    sorted_indices = np.argsort(arr1)
    arr1 = np.array(arr1, dtype='float64')[sorted_indices].tolist()
    arr2 = np.array(arr2, dtype='float64')[sorted_indices].tolist()
    new_arr1 = []
    new_arr2 = []
    while(len(arr1)>0):
      x1 = arr1[0]
      n = 0
      while(arr1[n]<=x1+0.01 and n+1 < len(arr1)):
        n+=1
      new_arr1.append(np.round(x1,2))
      new_arr2.append(np.round(np.mean(arr2[0:n+1]),2))
      if len(arr1)==n:
        break
      arr1 = arr1[n+1:]
      arr2 = arr2[n+1:]
    return new_arr1, new_arr2


#Returns the information to populate a graph of these two variables against each other
def fetch_data(x_axis, y_axis):
    if (x_axis == y_axis):
      df = pd.read_sql_query("SELECT "+col_expand(x_axis)+" from SongData inner join ProcessedSongData on SongData.SpotifyID = ProcessedSongData.SpotifyID", my_conn)
      val1 = df[x_axis].tolist()
      val2 = df[y_axis].tolist()
    else:
      df = pd.read_sql_query("SELECT "+col_expand(x_axis)+","+col_expand(y_axis)+" from SongData inner join ProcessedSongData on SongData.SpotifyID = ProcessedSongData.SpotifyID", my_conn)
      val1 = df[x_axis].tolist()

      #workaround type fix for sentiment - storing as string allowed with the intent of 
      #being able to expand the field to different sentiment values / custom types
      if x_axis == "Sentiment":
        for i in range(0, len(val1)):
          val1[i]=float(val1[i])
      val2 = df[y_axis].tolist()
      if y_axis == "Sentiment":
        for i in range(0, len(val2)):
          val2[i]=float(val2[i])

    if x_axis not in discrete_types:
        collapsed_x, collapsed_y = collapse_within_range(val1, val2)
        return ([collapsed_x, collapsed_y])
    else:
        collapsed_x, collapsed_y = collapse_by_equality(val1, val2)
        return ([collapsed_x, collapsed_y])



#Below section produces a string intended to paste directly into the code of the anvil app
#This bypasses all forms of paid data table access (despite making that file of the anvil code look terrible)
#String will look like the below code
#Can be auto-formatted in anvil as desired
nested_dict_formatting_example = {"x_type_1":{
    "y_type_1":
        [1,2,3]
,   "y_type_2":
        [7,8,9]
}
                                  
,"x_type_2":{
    "y_type_1":
        [4,5,6]
,   "y_type_2":
        [9,8,7]
}

}

#As described above, this produces a string that is intended to be copied and pasted into other python code
#It is essentially code to produce other code
#Returns a DIY python formatted dictionary of roughly the format above
#Any potential call to fetch_data(x_axis, y_axis) is replaced by a nested dictionary access DATA[x_axis][y_axis]
def nested_dictionary_cache_text():

    cache_strings_2d = []

    for x_type in all_x_types:
        cache_strings_1d = []
        for y_type in all_y_types:
            x_by_y_data = fetch_data(x_type, y_type)
            x_by_y_text = "["
            for data_point in x_by_y_data:
                x_by_y_text+=str(data_point) + ","
            x_by_y_text = x_by_y_text[:len(x_by_y_text)-1] #cut extraneous comma
            x_by_y_text += "]"
            cache_strings_1d.append(x_by_y_text)
        cache_strings_2d.append(cache_strings_1d)

    full_string = "{"
    for x in range(0, len(all_x_types)):
        x_type = all_x_types[x]
        dict_string_x = "\"" + x_type+ "\":{\n"

        for y in range(0,len(all_y_types)):
            y_type = all_y_types[y]
            dict_string_xy = "\t\"" + y_type + "\":\n\t\t"+ cache_strings_2d[x][y] + "\n,"
            dict_string_x += dict_string_xy
        dict_string_x = dict_string_x[:len(dict_string_x)-1] #cut extraneous comma
        dict_string_x += "}\n\n,"
        full_string += dict_string_x
    full_string = full_string[:len(full_string)-1] #cut extraneous comma
    full_string += "}\n\n"

    return full_string
    









