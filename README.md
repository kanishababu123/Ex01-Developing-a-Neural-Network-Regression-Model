# Ex01-Developing-a-Neural-Network-Regression-Model


{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import auth\n",
    "import gspread\n",
    "from google.auth import default\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Authenticate User"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "auth.authenticate_user()\n",
    "creds, _ = default()\n",
    "gc = gspread.authorize(creds)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Open the Google sheet and convert into DataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "worksheet = gc.open('StudentsData').sheet1\n",
    "\n",
    "rows = worksheet.get_all_values()\n",
    "\n",
    "df = pd.DataFrame(rows[1:], columns=rows[0])\n",
    "df = df.astype({'CGPA':'float'})"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
