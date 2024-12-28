// components/RotatingAtoms.tsx
import React from "react";
import { motion } from "framer-motion";

interface RotatingAtomProps {
  size: number; // size in pixels
  color: string; // Tailwind color class, e.g., 'bg-blue-500'
  orbitRadius: number; // distance from the center in pixels
  duration: number; // rotation duration in seconds
}

const RotatingAtoms: React.FC = () => {
  // Define the atoms with varying properties
   const atoms: RotatingAtomProps[] = [
     { size: 8, color: "bg-blue-500", orbitRadius: 60, duration: 12 },
     { size: 6, color: "bg-blue-400", orbitRadius: 80, duration: 8 },
     { size: 10, color: "bg-blue-600", orbitRadius: 100, duration: 16 },
     { size: 7, color: "bg-blue-300", orbitRadius: 120, duration: 10 },
     { size: 9, color: "bg-blue-700", orbitRadius: 140, duration: 14 },
     { size: 9, color: "bg-blue-300", orbitRadius: 160, duration: 10 },
     { size: 9, color: "bg-blue-900", orbitRadius: 180, duration: 14 },
     { size: 7, color: "bg-blue-500", orbitRadius: 50, duration: 9 },
     { size: 6, color: "bg-blue-400", orbitRadius: 70, duration: 7 },
     { size: 10, color: "bg-blue-600", orbitRadius: 90, duration: 15 },
     { size: 8, color: "bg-blue-500", orbitRadius: 110, duration: 13 },
     { size: 7, color: "bg-blue-700", orbitRadius: 130, duration: 17 },
     // Add more atoms as desired for further complexity
   ];


  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
      {atoms.map((atom, index) => (
        <motion.div
          key={index}
          className="absolute"
          animate={{ rotate: 360 }}
          transition={{
            repeat: Infinity,
            duration: atom.duration,
            ease: "linear",
          }}
        >
          <div
            className={`rounded-full ${atom.color} filter  opacity-80`}
            style={{
              width: atom.size,
              height: atom.size,
              position: "absolute",
              top: `calc(50% - ${atom.size / 2}px)`,
              left: `calc(50% - ${atom.size / 2}px)`,
              transform: `translate(${atom.orbitRadius}px, 0)`,
              boxShadow: `0 0 ${atom.size * 2}px ${atom.color.replace(
                "bg-",
                ""
              )}`,
            }}
          ></div>
          {/* Core Atom (optional): */}
          <div
            className={`rounded-full ${atom.color} opacity-100`}
            style={{
              width: atom.size / 2,
              height: atom.size / 2,
              position: "absolute",
              top: `calc(50% - ${atom.size / 4}px)`,
              left: `calc(50% - ${atom.size / 4}px)`,
            }}
          ></div>
        </motion.div>
      ))}
    </div>
  );
};

export default RotatingAtoms;
