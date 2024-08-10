def css():
    return '''
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #2b313e;
        flex-direction: row-reverse;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 30%; /* Adjust as needed */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .chat-message .avatar img {
        width: 100%;  /* Make the image cover the container */
        height: auto; /* Maintain aspect ratio */
        border-radius: 50%;
    }
    .chat-message .message {
        width: 70%; /* Adjust as needed */
        padding: 0 1rem;
        color: #fff;
        overflow-wrap: break-word;
    }
    </style>
    '''

def bot_template(message):
    return f'''
    <div class="chat-message bot">
        <div class="avatar">
            <img src="">
        </div>
        <div class="message">{message}</div>
    </div>
    '''

def user_template(message):
    return f'''
    <div class="chat-message user">
        <div class="avatar">
            <img src="">
        </div>    
        <div class="message">{message}</div>
    </div>
    '''
