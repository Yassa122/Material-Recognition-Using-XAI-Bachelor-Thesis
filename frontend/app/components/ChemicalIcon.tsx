// components/ChemicalIcon.tsx
"use client";
import React from "react";
import { motion } from "framer-motion";

interface ChemicalIconProps {
  icon: React.ReactNode;
  label: string;
}

const ChemicalIcon: React.FC<ChemicalIconProps> = ({ icon, label }) => {
  return (
    <motion.div
      className="flex flex-col items-center cursor-pointer"
      whileHover={{ scale: 1.2, rotate: 10 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className="w-16 h-16 flex justify-center items-center bg-gray-200 dark:bg-zinc-700 rounded-full mb-2">
        {icon}
      </div>
      <span className="text-gray-700 dark:text-zinc-200">{label}</span>
    </motion.div>
  );
};

export default ChemicalIcon;
