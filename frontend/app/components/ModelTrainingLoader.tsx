import Image from "next/image";
import StarIcon from "@/public/Star.svg"; // Adjust the path as needed

const ModelTrainingLoader = () => {
  return (
    <div className="bg-[#202020] p-16 rounded-lg shadow-lg flex flex-col items-center justify-center space-y-6 w-full h-full">
      {/* Animated SVG Icon */}
      <div className="bg-[#181818] p-6 rounded-full animate-pulse">
        <Image
          src={StarIcon}
          alt="Loading Icon"
          width={84}
          height={84}
          className="animate-spin-slow"
        />
      </div>

      {/* Title */}
      <h2 className="text-3xl font-bold text-white">Model Training...</h2>
      <p className="text-gray-400 text-lg">Your data is being loaded...</p>

      {/* Glowing and Animated Loading Bar */}
      <div className="w-full mt-28">
        <div className="h-4 rounded-full bg-gray-700 overflow-hidden relative mt-28">
          <div
            className="h-4 rounded-full absolute top-0 left-0 animate-gradient"
            style={{
              width: "100%",
              background: "linear-gradient(90deg, #7F00FF, #E100FF)",
              boxShadow: "0 0 10px #7F00FF, 0 0 20px #E100FF",
            }}
          ></div>
        </div>
      </div>

      <style jsx>{`
        .animate-spin-slow {
          animation: spin 4s linear infinite;
        }

        @keyframes spin {
          0% {
            transform: rotate(0deg);
          }
          100% {
            transform: rotate(360deg);
          }
        }

        .animate-gradient {
          animation: loading-gradient 2s linear infinite;
        }

        @keyframes loading-gradient {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
      `}</style>
    </div>
  );
};

export default ModelTrainingLoader;
