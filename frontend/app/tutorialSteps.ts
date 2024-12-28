// app/tutorialSteps.ts
import { ReactNode } from "react";

interface TutorialStep {
  selector: string; // CSS selector for the target element
  content: string | ReactNode; // Content to display in the tooltip
  title?: string; // Optional title for the tooltip

  position?: "top" | "right" | "bottom" | "left" | "auto";
}

const tutorialSteps: TutorialStep[] = [
  {
    selector: ".sidebar-dashboard",
    content:
      "This is the Dashboard button. Click here to view your main dashboard.",
    position: "right",
  },
  {
    selector: ".sidebar-data-upload",
    content:
      "Use this button to upload your data. Ensure your data is formatted correctly.",
    position: "right",
  },
  {
    selector: ".sidebar-model-training",
    content:
      "Start training your AI model here. Training status will be displayed on the dashboard.",
    position: "right",
  },
  {
    selector: ".header-search",
    content:
      "Use the search bar to quickly find features and data within the dashboard.",
    position: "bottom",
  },
  {
    selector: ".header-notifications",
    content:
      "Click here to view notifications and updates related to your account.",
    position: "bottom",
  },
  {
    selector: ".header-profile",
    content:
      "Access your profile and account settings by clicking on the profile icon.",
    position: "bottom",
  },
  {
    selector: ".start-training-button",
    content:
      "Initiate the training process by clicking the Start Training button.",
    position: "top",
  },
  {
    selector: "#sidebar", // ID of the Sidebar component
    title: "Sidebar Navigation",
    content:
      "Use the sidebar to navigate through different sections of the application.",
    position: "right",
  },
  {
    selector: "#header", // ID of the Header component
    title: "Header",
    content:
      "The header provides quick access to user settings and notifications.",
    position: "bottom",
  },
  {
    selector: "#upload-data", // ID of the UploadData component
    title: "Upload Data",
    content:
      "Upload your CSV files here to start training or predicting models.",
    position: "right",
  },
  {
    selector: "#storage-component", // ID of the StorageComponent
    title: "Storage Overview",
    content: "Monitor your data storage and manage uploaded files efficiently.",
    position: "left",
  },
  {
    selector: "#smiles-data-table", // ID of the SmilesDataTable
    title: "SMILES Data Table",
    content: "View and manage the parsed SMILES data from your uploaded files.",
    position: "top",
  },
  {
    selector: "#model-selection", // ID of the Model Selection dropdown
    title: "Model Selection",
    content:
      "Choose between different models and prediction methods for your data.",
    position: "bottom",
  },
  {
    selector: "#start-action-button", // ID of the Start Action button
    title: "Start Action",
    content:
      "Initiate training, prediction, or generation processes with this button.",
    position: "top",
  },
];

export default tutorialSteps;
