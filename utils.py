def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def add_to_results(event: dict, results: dict):
    message = event.get("messages")

    """"
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.content != "":
            results[message.type] = message.content
    """

    if message:
            if isinstance(message, list):
                message = message[-1]
            
            if message.content != "":
                # Track the query and what tool was used to process it
                step = {
                    "message_content": message.content,
                    "info": message
                }
                # Append the current step to the results dictionary using message type as the key
                results[message.type] = results.get(message.type, [])
                results[message.type].append(step)
