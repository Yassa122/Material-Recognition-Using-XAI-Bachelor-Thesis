// components/ChemicalBond.tsx
"use client";
import React from "react";
import { motion } from "framer-motion";

interface ChemicalBondProps {
  from: { x: number; y: number };
  to: { x: number; y: number };
}

const ChemicalBond: React.FC<ChemicalBondProps> = ({ from, to }) => {
  // Calculate the distance and angle between the two points
  const deltaX = to.x - from.x;
  const deltaY = to.y - from.y;
  const distance = Math.sqrt(deltaX ** 2 + deltaY ** 2);
  const angle = (Math.atan2(deltaY, deltaX) * 180) / Math.PI;

  return (
    <motion.div
      className="absolute bg-gray-300 dark:bg-zinc-600"
      style={{
        width: distance,
        height: 2, // Thickness of the bond
        transformOrigin: "0 0",
        transform: `rotate(${angle}deg)`,
        top: from.y,
        left: from.x,
        borderRadius: "1px",
      }}
      whileHover={{
        backgroundColor: "#3B82F6", // Tailwind Blue-500
        scaleY: 1.5,
        transition: { duration: 0.3 },
      }}
    />
  );
};

export default ChemicalBond;
