// components/Sidebar.tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image"; // Import for using SVG image
import StarsIcon from "@/public/stars.svg"; // Adjust path if needed
import {
  MdDashboard,
  MdCloudUpload,
  MdBarChart,
  MdSwapHoriz,
  MdLightbulb,
  MdSettings,
  MdPerson,
  MdLock,
} from "react-icons/md";

const SidebarLink = ({
  href,
  icon: Icon,
  label,
  customIcon, // Optional custom icon prop
}: {
  href: string;
  icon?: any;
  label: string;
  customIcon?: string;
}) => {
  const pathname = usePathname();
  const isActive = pathname === href;

  return (
    <Link href={href}>
      <div
        className={`relative group flex items-center space-x-4 py-4 rounded-md ${
          isActive
            ? "bg-gray-800 text-blue-400"
            : "hover:bg-gray-700 text-gray-300"
        }`}
      >
        {isActive && (
          <div className="absolute right-0 w-1 h-full bg-blue-500 rounded-full"></div>
        )}

        {/* Use custom icon if provided; otherwise, use Icon component */}
        {customIcon ? (
          <Image
            src={customIcon}
            alt={`${label} Icon`}
            width={24}
            height={24}
            className={`group-hover:text-white ${
              isActive ? "text-blue-400" : "text-gray-400"
            }`}
          />
        ) : (
          Icon && (
            <Icon
              className={`h-6 w-6 ${
                isActive ? "text-blue-400" : "text-gray-400"
              } group-hover:text-white`}
            />
          )
        )}
        <span
          className={`text-base font-semibold ${
            isActive ? "text-blue-400" : "text-gray-300"
          } group-hover:text-white`}
        >
          {label}
        </span>
      </div>
    </Link>
  );
};

const Sidebar = () => {
  return (
    <div className="w-64 bg-sidebarBg text-gray-100 pl-6 min-h-screen">
      <h2 className="text-2xl font-bold mb-10 text-center text-white">
        Explain Mat
      </h2>
      <hr className="border-gray-600 mb-8" />
      <nav className="space-y-4">
        <SidebarLink
          href="/pages/dashboard"
          icon={MdDashboard}
          label="Dashboard"
        />
        <SidebarLink
          href="/pages/data-upload"
          icon={MdCloudUpload}
          label="Data Upload"
        />

        {/* Custom SVG Icon for Model Training */}
        <SidebarLink
          href="/pages/model-training"
          label="Model Training"
          customIcon={StarsIcon} // Path to your SVG icon
        />

        <SidebarLink
          href="/pages/shap-explanations"
          icon={MdBarChart}
          label="SHAP Explanations"
        />
        <SidebarLink
          href="/dashboard/lime-explanations"
          icon={MdSwapHoriz}
          label="LIME Explanations"
        />
        <SidebarLink
          href="/dashboard/predictions"
          icon={MdLightbulb}
          label="Predictions"
        />
        <SidebarLink
          href="/dashboard/settings"
          icon={MdSettings}
          label="Settings"
        />
        <SidebarLink
          href="/dashboard/profile"
          icon={MdPerson}
          label="Profile"
        />
        <SidebarLink href="/sign-in" icon={MdLock} label="Sign In" />
      </nav>
    </div>
  );
};

export default Sidebar;
