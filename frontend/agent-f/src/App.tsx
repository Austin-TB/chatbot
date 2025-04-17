import React, {useState} from "react";

interface Message {
  role: 'system' | 'assistant' | 'user';
  content: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {role: 'system', content: 'You are a helpful AI assistant'}
  ]);
  const [input, setInput] = useState("");

  const handleSubmit = async (e:React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return;

    const userMessage : Message = {
      role: "user",
      content: input
    }

    setMessages(prev=>[...prev, userMessage])

    try {
      const response = await fetch('http://localhost:8888/chat/completions', 
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            messages: [...messages],
            temperature: 0.7,
            max_tokens: 128
          })
        })

        const data = await response.json();

        setMessages(prev=>[...prev, data.choices[0].message])
        setInput("");
    } catch(error) {
      console.log("error:",error);
    }
  }
  
  return (
    <div>
      <h1>Chat with AI</h1>
      
      {/* Display messages */}
      <div style={{margin: '20px 0'}}>
        {messages.filter(message=>message.role!=="system").map((message, index) => (
          <div key={index}>
            <strong>{message.role}:</strong> {message.content}
          </div>
        ))}
      </div>

      <form>
        <input 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
        />
        <button onClick ={handleSubmit}>send</button>
      </form>
    </div>
  );
}

export default App;