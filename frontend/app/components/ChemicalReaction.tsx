// components/ChemicalReaction.tsx
import React from "react";
import { FaFlask, FaLightbulb, FaWater } from "react-icons/fa"; // Example icons
import { motion } from "framer-motion";

interface ChemicalReactionProps {
  reaction: string;
  equation: string;
}

const ChemicalReaction: React.FC<ChemicalReactionProps> = ({
  reaction,
  equation,
}) => {
  // Choose an icon based on the reaction name or type
  const getIcon = (reactionName: string) => {
    if (reactionName.toLowerCase().includes("combustion")) {
      return <FaFlask size={30} className="text-orange-500" />;
    } else if (reactionName.toLowerCase().includes("photosynthesis")) {
      return <FaLightbulb size={30} className="text-orange-500" />;
    } else if (reactionName.toLowerCase().includes("formation of water")) {
      return <FaWater size={30} className="text-orange-500" />;
    } else {
      return <FaFlask size={30} className="text-orange-500" />; // Default icon
    }
  };

  return (
    <motion.div
      className="bg-white dark:bg-zinc-700 p-6 rounded-lg shadow-md flex flex-col items-center text-center transition-transform transform hover:scale-105"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      aria-label={`Chemical reaction: ${reaction}`}
    >
      {/* Icon */}
      <div className="mb-4">{getIcon(reaction)}</div>

      {/* Reaction Name */}
      <h3 className="text-xl font-semibold text-gray-900 dark:text-zinc-100 mb-2">
        {reaction}
      </h3>

      {/* Equation */}
      <p className="text-gray-600 dark:text-zinc-400">{equation}</p>
    </motion.div>
  );
};

export default ChemicalReaction;
