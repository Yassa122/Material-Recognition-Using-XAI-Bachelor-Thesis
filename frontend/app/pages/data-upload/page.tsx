// pages/data-upload.tsx
"use client";
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation"; // for page redirection
import Sidebar from "@/app/components/Sidebar";
import Header from "@/app/components/Header";
import UploadData from "@/app/components/UploadData";
import StorageComponent from "@/app/components/StorageComponent";
import SmilesDataTable from "@/app/components/SmilesDataTable";

// Update the URL of your Flask backend API here
const TRAIN_API_URL = "http://localhost:5000/train"; // Change to your Flask API URL

const DataUploadPage = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const router = useRouter(); // for page redirection

  // Handle the start training button click
  const handleStartTraining = async () => {
    const fileInput = document.querySelector<HTMLInputElement>("#fileInput"); // Assuming file input has an ID

    if (!fileInput?.files?.length) {
      setError("No file selected.");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    setLoading(true); // Start loading when the request begins
    setError(null); // Reset any previous errors
    setSuccessMessage(null); // Reset previous success messages

    try {
      const response = await fetch(TRAIN_API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setSuccessMessage("Model training started successfully!");
        setLoading(false);
        // Redirect to the model training page after success
        setTimeout(() => {
          router.push("/pages/model-training"); // Redirect after 2 seconds
        }, 2000);
      } else {
        setError(
          data.error || "An error occurred while starting the training."
        );
        setLoading(false);
      }
    } catch (err) {
      setError("Failed to connect to the server.");
      setLoading(false);
    }
  };

  return (
    <div className="bg-mainBg min-h-screen text-gray-100 flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 p-6">
        {/* Header */}
        <Header />

        {/* Page Title */}
        <h1 className="text-2xl font-bold mb-6">Data Upload</h1>

        {/* Components Container */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Storage Component */}
          <StorageComponent />

          {/* Upload Component */}
          <div className="bg-[#202020] p-6 rounded-lg shadow-lg flex flex-col justify-center items-center">
            <UploadData />
            <div className="text-gray-300 mt-4 flex flex-col items-center">
              <button
                onClick={handleStartTraining}
                disabled={loading}
                className="bg-blue-600 text-white py-2 px-6 rounded-lg mt-4"
              >
                {loading ? "Starting Training..." : "Start Training"}
              </button>
              {error && <p className="text-red-500 mt-2">{error}</p>}
              {successMessage && (
                <p className="text-green-500 mt-2">{successMessage}</p>
              )}
            </div>
          </div>
        </div>
        <SmilesDataTable />

        {/* Loading Modal */}
        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
            <div className="bg-white p-6 rounded-lg flex items-center">
              <svg
                className="animate-spin h-8 w-8 text-blue-600"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 1 1 16 0A8 8 0 0 1 4 12z"
                ></path>
              </svg>
              <p className="ml-4 text-gray-700">Loading... Please wait.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataUploadPage;
