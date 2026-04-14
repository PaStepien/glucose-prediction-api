from client import query_llama_classifier
from mock_data import mock_data
from assistant_service import detect_intent, handle_user_message


while True:
    message = input("You: ").strip()
    if message.lower() in {"exit", "quit", "q"}:
        print("Bye")
        break
    if not message:
        continue
    
    response = handle_user_message(
        message=message,
        data=mock_data
    )
    
    print(response)