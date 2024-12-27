// types/react-intro.js.d.ts

declare module "react-intro.js" {
  import { Component, ReactNode } from "react";

  interface IntroJsProps {
    steps: {
      element: string;
      intro: string;
      position?: string;
    }[];
    initialStep?: number;
    enabled?: boolean;
    onExit?: () => void;
    options?: {
      showProgress?: boolean;
      showBullets?: boolean;
      exitOnOverlayClick?: boolean;
      exitOnEsc?: boolean;
      nextLabel?: string;
      prevLabel?: string;
      skipLabel?: string;
      doneLabel?: string;
      tooltipPosition?: string;
      [key: string]: any; // Allow additional options
    };
  }

  export default class IntroJs extends Component<IntroJsProps> {}
}
