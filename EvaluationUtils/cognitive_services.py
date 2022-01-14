from Animator.utils import eprint
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os
import traceback
import http.client, urllib.request, urllib.parse, urllib.error, base64


def detect_text(image_path, language='ja'):
    region = os.environ['ACCOUNT_REGION']
    key = os.environ['ACCOUNT_KEY']

    credentials = CognitiveServicesCredentials(key)
    client = ComputerVisionClient(
        endpoint="https://" + region + ".api.cognitive.microsoft.com/",
        credentials=credentials
    )

    url = "https://github.com/Azure-Samples/cognitive-services-python-sdk-samples/raw/master/samples/vision/images/make_things_happen.jpg"
    numberOfCharsInOperationId = 36

    # SDK call
    rawHttpResponse = client.read(url, language=language, raw=True)

    # Get ID from returned headers
    operationLocation = rawHttpResponse.headers["Operation-Location"]
    idLocation = len(operationLocation) - numberOfCharsInOperationId
    operationId = operationLocation[idLocation:]

    # SDK call
    result = client.get_read_result(operationId)

    # Get data
    if result.status == OperationStatusCodes.succeeded:

        for line in result.analyze_result.read_results[0].lines:
            print(line.text)
            print(line.bounding_box)

    headers = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': '{subscription key}',
    }

    params = urllib.parse.urlencode({
        # Request parameters
        'language': f'{language}',
        'pages': '{string}',
    })

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    try:
        conn = http.client.HTTPSConnection('westus2.api.cognitive.microsoft.com')
        conn.request("POST", "/vision/v3.1-preview.2/read/analyze?%s" % params, f"{encoded_string}", headers)
        response = conn.getresponse()
        data = response.read()
        print(data)
        conn.close()
        return data
    except Exception as e:
        traceback.print_exc()
        eprint(' with exception: \'{}\'' % e)


if __name__ == '__main__':
    detect_text('image_path', language='ja')
