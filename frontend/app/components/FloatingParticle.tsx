// components/FloatingParticle.tsx
"use client";
import React from "react";
import { motion } from "framer-motion";

interface FloatingParticleProps {
  size: number;
  color: string;
  initialPosition: { x: number; y: number };
  hoverScale: number;
}

const FloatingParticle: React.FC<FloatingParticleProps> = ({
  size,
  color,
  initialPosition,
  hoverScale,
}) => {
  return (
    <motion.div
      className="absolute rounded-full pointer-events-none"
      style={{
        width: size,
        height: size,
        backgroundColor: color,
        top: initialPosition.y,
        left: initialPosition.x,
      }}
      animate={{
        y: [0, -20, 0],
        x: [0, 20, 0],
        rotate: [0, 360],
      }}
      transition={{
        duration: 10,
        repeat: Infinity,
        ease: "easeInOut",
      }}
      whileHover={{ scale: hoverScale }}
    />
  );
};

export default FloatingParticle;
