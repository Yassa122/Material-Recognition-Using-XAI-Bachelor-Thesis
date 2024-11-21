"use client";

import { useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { motion } from "framer-motion";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ChatComponent = () => {
  const [messages, setMessages] = useState([
    {
      sender: "system",
      text: `The model made this prediction based on the following features and their contributions:\n\n• Molecular Weight had a strong positive impact (+20%), meaning higher molecular weight increased the likelihood of the compound being classified as active.\n• Solubility contributed negatively (-15%), which decreased the likelihood of the compound being active.\n• pH Level had a moderate positive influence (+10%), indicating that certain pH levels are associated with active compounds.\n• Toxicity had a slight negative effect (-5%), reducing the prediction score slightly.\n• Chemical Group had a minor positive influence (+5%).\n\n**Conclusion:**\n\nThe model predicted that this compound is likely active with a high confidence level of 85%, largely driven by its molecular weight and pH level. However, the solubility and toxicity characteristics pulled the prediction slightly in the opposite direction.`,
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = () => {
    if (!input.trim()) return;

    // Append user message
    const newMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, newMessage]);

    // Simulate system typing
    setIsTyping(true);
    setTimeout(() => {
      const systemResponse = {
        sender: "system",
        text: "This output is influenced by the molecular weight and solubility due to their high SHAP values.",
      };
      setMessages((prev) => [...prev, systemResponse]);
      setIsTyping(false);
    }, 2000); // Simulate 2 seconds delay

    // Clear input
    setInput("");
  };

  // Bar Chart Data
  const barData = {
    labels: ["00", "04", "08", "12", "14", "16", "18"], // Example X-axis labels
    datasets: [
      {
        label: "Impact",
        data: [2, 1.5, 3, 2.7, 3.5, 4, 1], // Example data
        backgroundColor: "rgba(75, 192, 192, 0.6)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
        borderRadius: 5,
        barPercentage: 0.7,
      },
    ],
  };

  const barOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: true },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#9CA3AF" }, // Tailwind gray-400
      },
      y: {
        grid: { color: "#374151" }, // Tailwind gray-700
        ticks: { color: "#9CA3AF" }, // Tailwind gray-400
      },
    },
  };

  return (
    <div className="flex flex-col h-full bg-sidebarBg rounded-xl shadow-lg">
      {/* Chat Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center px-6 py-4 bg-black rounded-t-xl"
      >
        <div className="w-10 h-10 bg-chatIconColor rounded-full flex items-center justify-center text-white font-bold">
          Y
        </div>
        <div className="ml-3">
          <div className="text-sm text-gray-100 font-bold">You</div>
          <div className="text-xs text-gray-400">LIME Explanation:</div>
        </div>
      </motion.div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto bg-sidebarBg p-4 space-y-4">
        {messages.map((msg, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: idx * 0.2 }}
            className={`flex ${
              msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`${
                msg.sender === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-mainBg text-gray-200"
              } rounded-lg p-4 max-w-xl shadow-md`}
            >
              <div
                className={`${
                  msg.sender === "system" ? "text-sm font-semibold mb-2" : ""
                }`}
              >
                {msg.sender === "system" ? "LIME" : ""}
              </div>
              <p className="text-sm whitespace-pre-line">{msg.text}</p>
            </div>
          </motion.div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, repeat: Infinity }}
            className="flex items-center space-x-2"
          >
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
          </motion.div>
        )}

        {/* Chart Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="p-4 bg-zinc-900 rounded-lg mt-4 shadow-md"
        >
          <h3 className="text-sm font-bold text-gray-300 mb-2">
            LIME Bar Chart
          </h3>
          <div className="flex items-center justify-between mb-2">
            <p className="text-3xl font-semibold text-white">2.579</p>
            <span className="text-sm font-medium text-green-400">+2.45%</span>
          </div>
          <Bar data={barData} options={barOptions} />
        </motion.div>
      </div>

      {/* Input Field */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mt-4 flex items-center px-6 pb-4"
      >
        <input
          type="text"
          className="flex-1 bg-zinc-700 text-gray-200 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-black"
          placeholder="Message LIME"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <motion.button
          whileHover={{ scale: 1.2 }}
          whileTap={{ scale: 0.9 }}
          onClick={handleSend}
          className="ml-3 bg-zinc-500 flex items-center justify-center text-white w-12 h-12 rounded-lg hover:bg-zinc-900 focus:ring-2 focus:ring-black focus:outline-none"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 19V5m-7 7l7-7 7 7"
            />
          </svg>
        </motion.button>
      </motion.div>
    </div>
  );
};

export default ChatComponent;
