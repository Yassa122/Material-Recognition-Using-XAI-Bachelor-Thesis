// app/tutorialSteps.ts
import { ReactNode } from "react";

interface TutorialStep {
  selector: string; // CSS selector for the target element
  content: string | ReactNode; // Content to display in the tooltip
  position?: "top" | "right" | "bottom" | "left" | "auto";
}

const tutorialSteps: TutorialStep[] = [
  {
    selector: ".sidebar-dashboard",
    content: "This is the Dashboard button. Click here to view your main dashboard.",
    position: "right",
  },
  {
    selector: ".sidebar-data-upload",
    content: "Use this button to upload your data. Ensure your data is formatted correctly.",
    position: "right",
  },
  {
    selector: ".sidebar-model-training",
    content: "Start training your AI model here. Training status will be displayed on the dashboard.",
    position: "right",
  },
  {
    selector: ".header-search",
    content: "Use the search bar to quickly find features and data within the dashboard.",
    position: "bottom",
  },
  {
    selector: ".header-notifications",
    content: "Click here to view notifications and updates related to your account.",
    position: "bottom",
  },
  {
    selector: ".header-profile",
    content: "Access your profile and account settings by clicking on the profile icon.",
    position: "bottom",
  },
  {
    selector: ".start-training-button",
    content: "Initiate the training process by clicking the Start Training button.",
    position: "top",
  },
  // Add more steps as needed
];

export default tutorialSteps;
