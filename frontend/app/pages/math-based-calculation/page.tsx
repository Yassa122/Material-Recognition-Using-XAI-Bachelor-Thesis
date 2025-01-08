// pages/model-training/page.tsx
"use client";

import Sidebar from "@/app/components/Sidebar";
import MathBasedLoader from "@/app/components/MathBasedLoader";
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
          <MathBasedLoader taskId={""} statusEndpoint={""} onComplete={function (): void {
                      throw new Error("Function not implemented.");
                  } } />
        </div>
      </main>
    </div>
  );
};

export default ModelPredictingPage;
