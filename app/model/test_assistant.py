from mock_data import mock_data
from assistant_service import handle_user_message

response = handle_user_message(
    message="Predict my next glucose level?",
    data=mock_data
)

print(response)