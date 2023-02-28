# Python code to find the URL from an input string
import re

def Find(string):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))";
    res = [];
    url = re.findall(regex, string);
    res.append([x[0] for x in url]);
    return res


# Driver Code
company_name='Test'
string = 'My Profile: www.google.com https://auth.geeksforgseeks.org/user/Chinmoy%20Lenka/articles in the portal of https://www.geeksforgeeks.org/'
res=Find(string)
print(company_name, "  ", res)
