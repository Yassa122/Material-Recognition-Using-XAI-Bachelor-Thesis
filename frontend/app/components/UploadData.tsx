// components/UploadData.tsx
"use client";

import { useState } from "react";
import { FaUpload } from "react-icons/fa";

interface UploadDataProps {
  onFileSelected: (file: File) => void;
}

const UploadData: React.FC<UploadDataProps> = ({ onFileSelected }) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadError, setUploadError] = useState("");

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files ? e.target.files[0] : null;
    if (selectedFile) {
      setFile(selectedFile);
      onFileSelected(selectedFile); // Notify parent component
    }
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!file) {
      setUploadError("Please choose a file first!");
      return;
    }

    setUploading(true);
    setUploadSuccess(false);
    setUploadError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setUploadSuccess(true);
        setUploading(false);
      } else {
        throw new Error("Upload failed");
      }
    } catch (error) {
      setUploadError("Failed to upload file.");
      setUploading(false);
    }
  };

  return (
    <div className="flex bg-[#202020] p-6 rounded-lg shadow-lg space-x-8">
      {/* Upload Box */}
      <div className="flex-1 border-dashed border-2 bg-black border-gray-500 p-4 max-w-md w-full rounded-lg flex flex-col items-center justify-center relative">
        <input
          type="file"
          id="fileInput"
          name="file"
          accept=".csv,.json" // Restrict file types if needed
          className="absolute inset-0 opacity-0 cursor-pointer"
          onChange={handleFileChange}
        />
        <div className="flex flex-col items-center justify-center space-y-2 cursor-pointer">
          <FaUpload className="text-blue-500 text-4xl mb-2" />
          <p className="text-blue-400 font-semibold text-lg">Click to upload</p>
          <p className="text-gray-400 text-sm">CSV and JSON files allowed</p>
        </div>
        {file && (
          <div className="mt-2 text-white">
            <p>Selected file: {file.name}</p>
            <button
              className="mt-2 px-4 py-2 bg-blue-500 rounded-lg"
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? "Uploading..." : "Upload File"}
            </button>
          </div>
        )}
        {uploadError && <p className="text-red-500 mt-2">{uploadError}</p>}
        {uploadSuccess && (
          <p className="text-green-500 mt-2">Upload Successful!</p>
        )}
      </div>
    </div>
  );
};

export default UploadData;
