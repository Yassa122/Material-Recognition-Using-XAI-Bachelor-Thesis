"use client";

import React, { useEffect, useState, useRef } from "react";
import { createPortal } from "react-dom";
import { FaArrowLeft, FaArrowRight, FaTimes } from "react-icons/fa";
import tutorialSteps from "@/app/tutorialSteps";
import { motion, AnimatePresence } from "framer-motion";
import classNames from "classnames";

interface CustomTourProps {
  steps: typeof tutorialSteps;
  isOpen: boolean;
  onClose: () => void;
  darkMode: boolean;
}

const CustomTour: React.FC<CustomTourProps> = ({
  steps,
  isOpen,
  onClose,
  darkMode,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);
  const [isMounted, setIsMounted] = useState(false);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Scroll to target element
  const scrollToElement = (element: HTMLElement) => {
    element.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      setCurrentStep(0);
    } else {
      setIsMounted(false);
    }
  }, [isOpen]);

  useEffect(() => {
    if (isMounted && currentStep < steps.length) {
      const step = steps[currentStep];
      const element = document.querySelector(step.selector) as HTMLElement;

      if (element) {
        const rect = element.getBoundingClientRect();
        setTargetRect(rect);
        element.classList.add("tour-highlight");

        // Auto-scroll if the element is not in viewport
        const isInViewport =
          rect.top >= 0 &&
          rect.left >= 0 &&
          rect.bottom <=
            (window.innerHeight || document.documentElement.clientHeight) &&
          rect.right <=
            (window.innerWidth || document.documentElement.clientWidth);

        if (!isInViewport) {
          scrollToElement(element);
        }
      } else {
        console.warn(`Element not found for selector: ${step.selector}`);
        // Skip to next step after a short delay
        const timer = setTimeout(() => {
          setCurrentStep((prev) => prev + 1);
        }, 1000);
        return () => clearTimeout(timer);
      }

      // Cleanup: Remove highlight when step changes or tour closes
      return () => {
        if (element) {
          element.classList.remove("tour-highlight");
        }
      };
    }
  }, [isMounted, currentStep, steps]);

  useEffect(() => {
    const handleResizeOrScroll = () => {
      if (isMounted && currentStep < steps.length) {
        const step = steps[currentStep];
        const element = document.querySelector(step.selector) as HTMLElement;
        if (element) {
          const rect = element.getBoundingClientRect();
          setTargetRect(rect);
        }
      }
    };

    window.addEventListener("resize", handleResizeOrScroll);
    window.addEventListener("scroll", handleResizeOrScroll);

    return () => {
      window.removeEventListener("resize", handleResizeOrScroll);
      window.removeEventListener("scroll", handleResizeOrScroll);
    };
  }, [isMounted, currentStep, steps]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((prev) => prev + 1);
    } else {
      handleClose();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const handleClose = () => {
    onClose();
    setCurrentStep(0);
  };

  if (!isMounted || currentStep >= steps.length) {
    return null;
  }

  const step = steps[currentStep];

  // Calculate tooltip position
  const offset = 10; // Distance between element and tooltip
  let tooltipPosition: { top: number; left: number } = { top: 0, left: 0 };

  if (targetRect) {
    switch (step.position) {
      case "top":
        tooltipPosition = {
          top: targetRect.top - offset,
          left: targetRect.left + targetRect.width / 2,
        };
        break;
      case "right":
        tooltipPosition = {
          top: targetRect.top + targetRect.height / 2,
          left: targetRect.left + targetRect.width + offset,
        };
        break;
      case "bottom":
        tooltipPosition = {
          top: targetRect.bottom + offset,
          left: targetRect.left + targetRect.width / 2,
        };
        break;
      case "left":
        tooltipPosition = {
          top: targetRect.top + targetRect.height / 2,
          left: targetRect.left - offset,
        };
        break;
      default:
        tooltipPosition = {
          top: targetRect.bottom + offset,
          left: targetRect.left + targetRect.width / 2,
        };
    }

    // Adjust for viewport boundaries
    const tooltipWidth = 300; // Adjust as needed
    const tooltipHeight = 150; // Adjust as needed

    if (tooltipPosition.left - tooltipWidth / 2 < 10) {
      tooltipPosition.left = tooltipWidth / 2 + 10;
    } else if (
      tooltipPosition.left + tooltipWidth / 2 >
      window.innerWidth - 10
    ) {
      tooltipPosition.left = window.innerWidth - tooltipWidth / 2 - 10;
    }

    if (tooltipPosition.top - tooltipHeight < 10) {
      tooltipPosition.top = targetRect.bottom + offset;
    }
  }

  const tooltipStyle: React.CSSProperties = {
    position: "absolute",
    top: tooltipPosition.top + window.scrollY,
    left: tooltipPosition.left + window.scrollX,
    transform: "translate(-50%, -100%)",
    width: "300px",
    zIndex: 1000,
  };

  return createPortal(
    <AnimatePresence>
      {isMounted && (
        <>
          {/* Overlay */}
          <motion.div
            key="overlay"
            className="fixed inset-0 bg-black bg-opacity-50 z-40"
            onClick={handleClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            exit={{ opacity: 0 }}
            aria-hidden="true"
          ></motion.div>

          {/* Highlighted Element Overlay */}
          {targetRect && (
            <div
              className="fixed top-0 left-0 w-full h-full pointer-events-none z-50"
              style={{
                boxShadow: `
                  0 0 0 3px ${darkMode ? "#4F46E5" : "#3B82F6"},
                  0 0 10px rgba(0,0,0,0.5)
                `,
                borderRadius: "8px",
                top: targetRect.top + window.scrollY - 5,
                left: targetRect.left + window.scrollX - 5,
                width: targetRect.width + 10,
                height: targetRect.height + 10,
              }}
            ></div>
          )}

          {/* Tooltip */}
          {targetRect && (
            <motion.div
              key="tooltip"
              className="fixed z-50"
              style={tooltipStyle}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ duration: 0.3 }}
              ref={tooltipRef}
              role="dialog"
              aria-modal="true"
              aria-labelledby="tour-tooltip-title"
            >
              <div
                className={`bg-zinc-700 dark:bg-gray-800 p-4 rounded-lg shadow-lg ${
                  darkMode ? "text-gray-200" : "text-gray-200"
                }`}
              >
                {/* Content */}
                <div className="flex flex-col">
                  <h3
                    id="tour-tooltip-title"
                    className="text-lg font-semibold mb-2"
                  >
                    Step {currentStep + 1} of {steps.length}
                  </h3>
                  <div className="flex-1">{step.content}</div>

                  {/* Navigation Buttons */}
                  <div className="mt-4 flex justify-between items-center">
                    <button
                      onClick={handlePrevious}
                      disabled={currentStep === 0}
                      className={classNames(
                        "flex items-center px-3 py-1 rounded",
                        {
                          "bg-gray-300 text-gray-700 cursor-not-allowed":
                            currentStep === 0,
                          "bg-gray-200 text-gray-900 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-200":
                            currentStep > 0,
                        }
                      )}
                      aria-label="Previous Step"
                    >
                      <FaArrowLeft className="mr-1" /> Previous
                    </button>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm">
                        {currentStep + 1} / {steps.length}
                      </span>
                      <button
                        onClick={handleNext}
                        className="flex items-center px-3 py-1 bg-blue-500 text-gray-900 rounded hover:bg-blue-600"
                        aria-label={
                          currentStep === steps.length - 1
                            ? "Finish Tour"
                            : "Next Step"
                        }
                      >
                        {currentStep === steps.length - 1 ? "Finish" : "Next"}
                        <FaArrowRight className="ml-1" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </>
      )}
    </AnimatePresence>,
    document.body
  );
};

export default CustomTour;
