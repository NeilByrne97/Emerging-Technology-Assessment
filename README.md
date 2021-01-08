**Emerging-Technology-Assessment** 

**To Run App**

set FLASK_APP=webService.py
python -m flask run

docker build . -t webservice-image
docker run --name webservice-container -d -p 5000:5000 webservice-image
