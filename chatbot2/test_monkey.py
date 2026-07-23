import urllib.request
_orig_Request = urllib.request.Request
class MyRequest(_orig_Request):
    def __init__(self, *args, **kwargs):
        headers = kwargs.get('headers', {})
        if 'User-Agent' not in headers and 'user-agent' not in {k.lower() for k in headers}:
            headers['User-Agent'] = 'Mozilla/5.0'
        kwargs['headers'] = headers
        super().__init__(*args, **kwargs)
urllib.request.Request = MyRequest

from urllib import request as urlrequest
req = urlrequest.Request("https://api.groq.com/openai/v1/models")
print(req.headers)
