"use client";

import React, { useState, useEffect } from "react";
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
      text: "LIME and SHAP explanations will load here once processed.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [limeData, setLimeData] = useState([]);
  const [shapData, setShapData] = useState([]);
  const [loadingLime, setLoadingLime] = useState(true);
  const [loadingShap, setLoadingShap] = useState(true);

  // Fetch LIME and SHAP explanation data from the API
  useEffect(() => {
    const fetchLimeData = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/lime_status");
        const data = await response.json();

        if (data.status === "completed") {
          // Parse the LIME explanation weights into chart-compatible format
          const barChartData = data.result.weights.map(([feature, weight]) => ({
            label: feature,
            value: Math.abs(weight), // Use absolute values for bar height
          }));

          // Update state with the fetched LIME data
          setLimeData(barChartData);

          // Append a new message explaining the LIME output
          const explanationMessage = {
            sender: "system",
            text: `The LIME explanation shows the following feature impacts:\n\n${data.result.weights
              .map(
                ([feature, weight]) =>
                  `• ${feature}: ${weight > 0 ? "+" : ""}${(
                    weight * 100
                  ).toFixed(2)}% impact`
              )
              .join("\n")}\n\nKey features are displayed in the chart below.`,
          };
          setMessages((prev) => [...prev, explanationMessage]);
          setLoadingLime(false);
        } else if (data.status === "error") {
          setMessages((prev) => [
            ...prev,
            { sender: "system", text: "Error fetching LIME explanation." },
          ]);
          setLoadingLime(false);
        } else {
          setMessages((prev) => [
            ...prev,
            {
              sender: "system",
              text: "LIME explanation is still processing...",
            },
          ]);
        }
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          { sender: "system", text: "Failed to fetch LIME explanation." },
        ]);
      }
    };

    const fetchShapData = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/shap_status");
        const data = await response.json();

        if (data.status === "completed") {
          // Process SHAP data
          const shapResult = data.result;
          const features = shapResult.features;
          const shapValues = shapResult.shap_values[0][0]; // Assuming first sample

          // Structure data for visualization
          const shapChartData = features.map((feature, index) => ({
            label: feature,
            value: shapValues[index],
          }));

          setShapData(shapChartData);

          // Append a new message explaining the SHAP output
          const shapExplanationMessage = {
            sender: "system",
            text: `The SHAP explanation provides detailed insights into feature impacts for the prediction:\n\n${features
              .map(
                (feature, index) =>
                  `• ${feature}: ${shapValues[index] > 0 ? "+" : ""}${(
                    shapValues[index] * 100
                  ).toFixed(2)}% impact`
              )
              .join("\n")}\n\nKey features are displayed in the chart below.`,
          };
          setMessages((prev) => [...prev, shapExplanationMessage]);
          setLoadingShap(false);
        } else if (data.status === "error") {
          setMessages((prev) => [
            ...prev,
            { sender: "system", text: "Error fetching SHAP explanation." },
          ]);
          setLoadingShap(false);
        } else {
          setMessages((prev) => [
            ...prev,
            {
              sender: "system",
              text: "SHAP explanation is still processing...",
            },
          ]);
        }
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          { sender: "system", text: "Failed to fetch SHAP explanation." },
        ]);
      }
    };

    // Fetch both LIME and SHAP data when the component mounts
    fetchLimeData();
    fetchShapData();
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;

    // Append user message
    const newMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, newMessage]);

    // Simulate system typing
    setIsTyping(true);

    try {
      // Call Flask endpoint for OpenAI chat
      const response = await fetch("http://127.0.0.1:5000/api/chatgpt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [
            ...messages.map((msg) => ({
              role: msg.sender === "user" ? "user" : "system",
              content: msg.text,
            })),
            { role: "user", content: input },
          ],
          limeData, // Send LIME data as context
          shapData, // Send SHAP data as context
        }),
      });

      const data = await response.json();

      if (data.response) {
        // Append the API response to the chat
        const systemResponse = { sender: "system", text: data.response };
        setMessages((prev) => [...prev, systemResponse]);
      } else {
        setMessages((prev) => [
          ...prev,
          { sender: "system", text: "No response from OpenAI API." },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "system",
          text: "There was an error connecting to the API. Please try again.",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevents default behavior if necessary
      handleSend();
      setInput(""); // Clear input after sending the message
    }
  };

  // Bar Chart Data for LIME
  const limeBarData = {
    labels: limeData.map((d) => d.label),
    datasets: [
      {
        label: "Feature Impact",
        data: limeData.map((d) => d.value),
        backgroundColor: "rgba(75, 192, 192, 0.6)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
        borderRadius: 5,
        barPercentage: 0.7,
      },
    ],
  };

  const limeBarOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: true },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#9CA3AF" },
      },
      y: {
        grid: { color: "#374151" },
        ticks: { color: "#9CA3AF" },
      },
    },
  };

  // Bar Chart Data for SHAP
  const shapChartData = {
    labels: shapData.map((d) => d.label),
    datasets: [
      {
        label: "SHAP Value",
        data: shapData.map((d) => d.value),
        backgroundColor: shapData.map(
          (d) =>
            d.value > 0
              ? "rgba(75, 192, 192, 0.6)" // Positive impact
              : "rgba(255, 99, 132, 0.6)" // Negative impact
        ),
        borderColor: shapData.map((d) =>
          d.value > 0 ? "rgba(75, 192, 192, 1)" : "rgba(255, 99, 132, 1)"
        ),
        borderWidth: 1,
        borderRadius: 5,
        barPercentage: 0.7,
      },
    ],
  };

  const shapOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: true },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#9CA3AF" },
      },
      y: {
        grid: { color: "#374151" },
        ticks: { color: "#9CA3AF" },
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
          <div className="text-xs text-gray-400">LIME & SHAP Explanations:</div>
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

        {/* LIME Chart Section */}
        {!loadingLime && limeData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="p-4 bg-zinc-900 rounded-lg mt-4 shadow-md"
          >
            <h3 className="text-sm font-bold text-gray-300 mb-2">
              LIME Feature Impact Chart
            </h3>
            <Bar data={limeBarData} options={limeBarOptions} />
          </motion.div>
        )}

        {/* SHAP Chart Section */}
        {!loadingShap && shapData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="p-4 bg-zinc-900 rounded-lg mt-4 shadow-md"
          >
            <h3 className="text-sm font-bold text-gray-300 mb-2">
              SHAP Feature Impact Chart
            </h3>
            <Bar data={shapChartData} options={shapOptions} />
          </motion.div>
        )}
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
          placeholder="Message LIME & SHAP"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown} // Added onKeyDown handler
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
