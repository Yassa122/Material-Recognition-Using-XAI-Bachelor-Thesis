// app/data-upload/page.tsx
"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Papa from "papaparse";
import Sidebar from "@/app/components/Sidebar";
import Header from "@/app/components/Header";
import UploadData from "@/app/components/UploadData";
import StorageComponent from "@/app/components/StorageComponent";
import SmilesDataTable from "@/app/components/SmilesDataTable";
import { Switch } from "@headlessui/react";
import { FaTrain, FaBrain, FaCalculator, FaCogs } from "react-icons/fa";
import StaggeredDropDown from "@/app/components/StaggeredDropDown";
import CustomTour from "@/app/components/CustomTour";
import tutorialSteps from "@/app/tutorialSteps";

const TRAIN_API_URL = "http://localhost:5000/train";
const PREDICT_API_URL = "http://localhost:5000/predict";
const MATH_PREDICT_API_URL = "http://localhost:5000/calculate_properties_async";
const GENERATIVE_API_URL = "http://localhost:5000/generate"; // New Generative API

interface SMILESDatum {
  name: string;
  smiles: string;
  molecularWeight?: string;
  meltingPoint?: string;
  dateAdded?: string;
  [key: string]: any; // For dynamic properties
}

const DataUploadPage = () => {
  // Existing States
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isPredictMode, setIsPredictMode] = useState(false);
  const [predictionMethod, setPredictionMethod] = useState<
    "transformer" | "math"
  >("transformer");

  const [selectedModel, setSelectedModel] =
    useState<string>("Predictive Model"); // Updated State

  const [taskId, setTaskId] = useState<string | null>(null);
  const [calculationStatus, setCalculationStatus] = useState<string | null>(
    null
  );
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [pollingIntervalId, setPollingIntervalId] =
    useState<NodeJS.Timeout | null>(null);

  const [smilesData, setSmilesData] = useState<SMILESDatum[]>([]); // State to hold parsed CSV data

  const router = useRouter();
  const searchParams = useSearchParams();

  // Reference to the "Train" button
  const trainButtonRef = useRef<HTMLButtonElement>(null);

  // Tour States
  const [isTourOpen, setIsTourOpen] = useState(false); // State to control tour visibility

  // For demonstration, using local darkMode state
  const [darkMode, setDarkMode] = useState(false); // Add your logic to manage darkMode

  // Handle file selection from UploadData component
  const handleFileSelected = (file: File) => {
    // Parse CSV file using Papa Parse
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: function (results) {
        const parsedData = results.data as SMILESDatum[];
        setSmilesData(parsedData);
        setSuccessMessage("File parsed successfully!");
        setError(null);
      },
      error: function (err) {
        setError("Failed to parse CSV file.");
        setSuccessMessage(null);
      },
    });
  };

  const handleStartAction = async () => {
    const fileInput = document.querySelector<HTMLInputElement>("#fileInput");

    if (!fileInput?.files?.length) {
      setError("No file selected.");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    setLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      let apiUrl = TRAIN_API_URL;

      if (isPredictMode) {
        if (selectedModel === "Predictive Model") {
          if (predictionMethod === "transformer") {
            apiUrl = PREDICT_API_URL;
          } else {
            // For Math-Based Approach
            apiUrl = MATH_PREDICT_API_URL;
          }
        } else if (selectedModel === "Generative Model") {
          apiUrl = GENERATIVE_API_URL;
        }
      }

      const response = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        if (selectedModel === "Generative Model") {
          // Handle Generative Model Response
          setSuccessMessage("Generation started successfully!");
          // Implement any additional handling, such as storing generated data or providing download links
          // Example: If backend returns a URL to generated content
          if (data.generated_url) {
            setDownloadUrl(`http://localhost:5000${data.generated_url}`);
          }
        } else if (predictionMethod === "math") {
          // Store the task_id and start polling
          const { task_id } = data;
          setTaskId(task_id);
          setSuccessMessage("Calculation started successfully!");

          // Start polling the status endpoint
          const intervalId = setInterval(() => {
            checkCalculationStatus(task_id);
          }, 5000); // Poll every 5 seconds

          setPollingIntervalId(intervalId);
        } else {
          setSuccessMessage(
            isPredictMode
              ? "Predictions started successfully!"
              : "Model training started successfully!"
          );
          // Optionally, redirect based on action
          setTimeout(() => {
            router.push(
              isPredictMode
                ? "/pages/model-predicting"
                : "/pages/model-training"
            );
          }, 2000);
        }
      } else {
        setError(data.error || "An error occurred during the operation.");
      }
    } catch (err: any) {
      setError("Failed to connect to the server.");
    } finally {
      setLoading(false);
    }
  };

  const checkCalculationStatus = async (taskId: string) => {
    try {
      const response = await fetch(
        `http://localhost:5000/calculate_properties_status/${taskId}`
      );
      const data = await response.json();

      if (response.ok) {
        setCalculationStatus(data.status);

        if (data.status === "completed") {
          setSuccessMessage("Calculation completed successfully!");
          setDownloadUrl(`http://localhost:5000${data.download_url}`);

          // Stop polling
          if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            setPollingIntervalId(null);
          }
        } else if (data.status === "error") {
          setError(
            data.error_detail || "An error occurred during calculation."
          );
          // Stop polling
          if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            setPollingIntervalId(null);
          }
        } else {
          // Update progress message if available
          setSuccessMessage(data.message);
        }
      } else {
        setError(data.error || "Failed to get calculation status.");
        // Stop polling
        if (pollingIntervalId) {
          clearInterval(pollingIntervalId);
          setPollingIntervalId(null);
        }
      }
    } catch (err) {
      setError("Failed to connect to the server.");
      // Stop polling
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
        setPollingIntervalId(null);
      }
    }
  };

  // Clear polling interval on component unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
      }
    };
  }, [pollingIntervalId]);

  // Optional: Automatically start the tour for first-time users
  useEffect(() => {
    const hasCompletedTour = localStorage.getItem("hasCompletedTour");

    if (!hasCompletedTour) {
      setIsTourOpen(true);
      localStorage.setItem("hasCompletedTour", "true");
    }
  }, []);

  // Handle automatic triggering of the "Train" button based on query parameter
  useEffect(() => {
    const autoToggle = searchParams.get("autoToggle");

    if (autoToggle === "true" && trainButtonRef.current) {
      trainButtonRef.current.click();

      // Remove the query parameter to prevent re-triggering on refresh
      router.replace("/pages/data-upload");
    }
  }, [searchParams, router]);

  return (
    <div className="bg-mainBg min-h-screen text-gray-100 flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 ">
        {/* Header */}
        <Header />
        <div className="p-6">
          {/* Page Title */}
          <h1 className="text-3xl font-extrabold mb-6">Data Upload</h1>

          {/* Dropdown Menu for Model Type Selection */}
          <div className="mb-6">
            <StaggeredDropDown
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
            />
          </div>

          {/* Toggle Switch */}
          <div className="flex items-center mb-8">
            <span className="mr-4 text-lg font-medium">
              {isPredictMode ? "Predict Mode" : "Train Mode"}
            </span>
            <Switch
              checked={isPredictMode}
              onChange={setIsPredictMode}
              className={`${
                isPredictMode ? "bg-blue-600" : "bg-gray-300"
              } relative inline-flex h-8 w-16 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500`}
            >
              <span
                className={`${
                  isPredictMode ? "translate-x-8" : "translate-x-0"
                } inline-block h-6 w-6 transform rounded-full bg-white shadow-md transition-transform`}
              />
            </Switch>
          </div>

          {/* Prediction Method Selection (Only in Predict Mode and Predictive Model) */}
          {isPredictMode && selectedModel === "Predictive Model" && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold mb-4">
                Select Prediction Method:
              </h2>
              <div className="flex flex-col md:flex-row items-center md:items-stretch md:justify-start gap-4">
                {/* Transformer Model Card */}
                <div
                  onClick={() => setPredictionMethod("transformer")}
                  className={`flex flex-col items-center p-6 rounded-lg cursor-pointer shadow-md transition transform hover:scale-105 ${
                    predictionMethod === "transformer"
                      ? "bg-blue-600 text-white"
                      : "bg-[#202020] text-gray-300"
                  }`}
                >
                  <FaBrain className="text-4xl mb-4" />
                  <h3 className="text-lg font-semibold mb-2">
                    Transformer Model
                  </h3>
                  <p className="text-sm text-center">
                    Utilize advanced machine learning algorithms for prediction.
                  </p>
                </div>

                {/* Math-Based Approach Card */}
                <div
                  onClick={() => setPredictionMethod("math")}
                  className={`flex flex-col items-center p-6 rounded-lg cursor-pointer shadow-md transition transform hover:scale-105 ${
                    predictionMethod === "math"
                      ? "bg-blue-600 text-white"
                      : "bg-[#202020] text-gray-300"
                  }`}
                >
                  <FaCalculator className="text-4xl mb-4" />
                  <h3 className="text-lg font-semibold mb-2">
                    Math-Based Approach
                  </h3>
                  <p className="text-sm text-center">
                    Calculate properties using mathematical formulas and
                    descriptors.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Components Container */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Storage Component */}
            <StorageComponent />

            {/* Upload Component */}
            <div className="bg-[#202020] p-8 rounded-xl shadow-2xl flex flex-col justify-center items-center">
              <UploadData onFileSelected={handleFileSelected} />
              <div className="text-gray-300 mt-6 flex flex-col items-center w-full">
                <button
                  ref={trainButtonRef}
                  onClick={handleStartAction}
                  disabled={loading}
                  className={`flex items-center justify-center w-full bg-gradient-to-r from-blue-500 to-blue-700 text-white py-3 px-6 rounded-lg mt-4 shadow-lg hover:from-blue-600 hover:to-blue-800 transition-all duration-300 disabled:opacity-50`}
                  aria-label={
                    isPredictMode ? "Start Prediction" : "Start Training"
                  }
                >
                  {loading ? (
                    <>
                      <div className="animate-spin h-5 w-5 mr-3 border-2 border-white border-t-transparent rounded-full"></div>
                      {isPredictMode ? "Processing..." : "Training..."}
                    </>
                  ) : (
                    <>
                      {isPredictMode ? (
                        selectedModel === "Generative Model" ? (
                          <FaCogs className="mr-2" /> // Icon for Generative Model
                        ) : predictionMethod === "transformer" ? (
                          <FaBrain className="mr-2" />
                        ) : (
                          <FaCalculator className="mr-2" />
                        )
                      ) : (
                        <FaTrain className="mr-2" />
                      )}
                      {isPredictMode
                        ? selectedModel === "Generative Model"
                          ? "Start Generation"
                          : "Start Prediction"
                        : "Start Training"}
                    </>
                  )}
                </button>
                {error && <p className="text-red-500 mt-3">{error}</p>}
                {successMessage && (
                  <p className="text-green-500 mt-3">{successMessage}</p>
                )}
                {/* Download Link */}
                {downloadUrl && (
                  <div className="mt-4">
                    <a
                      href={downloadUrl}
                      download
                      className="text-blue-500 underline"
                    >
                      Download Results
                    </a>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Display SMILES Data Table */}
          {smilesData.length > 0 && <SmilesDataTable data={smilesData} />}

          {/* Loading Modal */}
          {loading && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
              <div className="bg-white p-6 rounded-lg flex items-center shadow-xl">
                <svg
                  className="animate-spin h-10 w-10 text-blue-600"
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
                    d="M4 12a8 8 0 018-8v8H4z"
                  ></path>
                </svg>
                <p className="ml-4 text-lg text-gray-700">
                  {isPredictMode
                    ? selectedModel === "Generative Model"
                      ? "Generating... Please wait."
                      : "Processing... Please wait."
                    : "Training... Please wait."}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* CustomTour Component */}
        <CustomTour
          steps={tutorialSteps}
          isOpen={isTourOpen}
          onClose={() => setIsTourOpen(false)}
          darkMode={darkMode} // Pass the darkMode state or prop
        />
      </div>
    </div>
  );
};

export default DataUploadPage;
