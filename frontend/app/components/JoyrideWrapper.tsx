// app/components/JoyrideWrapper.tsx
"use client";

import React from "react";
import Joyride, { CallBackProps, STATUS, Step } from "react-joyride";

interface JoyrideWrapperProps {
  steps: Step[];
  run: boolean;
  setRun: (run: boolean) => void;
  darkMode: boolean;
}

const JoyrideWrapper: React.FC<JoyrideWrapperProps> = ({
  steps,
  run,
  setRun,
  darkMode,
}) => {
  const handleJoyrideCallback = (data: CallBackProps) => {
    const { status } = data;
    const finishedStatuses: string[] = [STATUS.FINISHED, STATUS.SKIPPED];

    if (finishedStatuses.includes(status)) {
      setRun(false);
      localStorage.setItem("hasCompletedTutorial", "true");
    }
  };

  return (
    <Joyride
      steps={steps}
      run={run}
      continuous={true}
      showProgress={true}
      showSkipButton={true}
      styles={{
        options: {
          backgroundColor: darkMode ? "#1F2937" : "#FFFFFF", // Dark or light mode background
          textColor: darkMode ? "#FFFFFF" : "#000000", // Text color for readability
          primaryColor: "#4F46E5", // Tailwind's indigo-600
          overlayColor: "rgba(0, 0, 0, 0.5)", // Overlay color
          zIndex: 10000, // Ensure it's on top
          arrowColor: darkMode ? "#1F2937" : "#FFFFFF", // Match background color
          width: 300, // Set a fixed width for the tooltip
        },
        buttonNext: {
          backgroundColor: "#4F46E5",
          color: "#FFFFFF",
        },
        buttonBack: {
          backgroundColor: "#6B7280",
          color: "#FFFFFF",
        },
        buttonClose: {
          color: "#6B7280",
        },
      }}
      callback={handleJoyrideCallback}
      locale={{
        last: "Done",
        skip: "Skip",
        close: "Close",
        next: "Next",
        back: "Back",
      }}
    />
  );
};

export default JoyrideWrapper;
