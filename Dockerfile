FROM python:3.10-slim

ARG RUN_ID

CMD echo "Building Docker for RUN_ID=$RUN_ID"
