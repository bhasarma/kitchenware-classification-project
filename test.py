import requests

#use the below url if your are testing locally
#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

#for aws lamda deplyment. Below is the url we got from API Gateway
url = 'https://c8m95kkqv5.execute-api.us-east-1.amazonaws.com/test/predict' 

data = {'url': 'https://github.com/bhasarma/kitchenware-classification-project/blob/main/test-image.jpg?raw=true'}


result = requests.post(url, json=data).json()
print(result)