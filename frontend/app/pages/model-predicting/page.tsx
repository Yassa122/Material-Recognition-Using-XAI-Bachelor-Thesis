// pages/model-training/page.tsx
"use client";

import Sidebar from "@/app/components/Sidebar";
import ModelTrainingLoader from "@/app/components/ModelPredictingLoader";
import Header from "@/app/components/Header"; // Assume you have a Header component similar to what you've designed

const ModelPredictingPage = () => {
  return (
    <div className="bg-mainBg min-h-screen flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        <Header />
        <div className="flex items-center justify-center flex-1 p-4">
          <ModelTrainingLoader />
        </div>
      </main>
    </div>
  );
};

export default ModelPredictingPage;
