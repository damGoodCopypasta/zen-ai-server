FROM python:3.9
COPY ./ ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD ["litestar", "run", "--host" , "0.0.0.0"]