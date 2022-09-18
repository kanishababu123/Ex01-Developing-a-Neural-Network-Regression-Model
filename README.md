# Ex01-Developing-a-Neural-Network-Regression-Model


from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

 Authenticate User

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

 Open the Google sheet and convert into DataFrame

worksheet = gc.open('StudentsData').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'CGPA':'float'})
