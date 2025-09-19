"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "../ui/button";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { Sheet, SheetContent, SheetTrigger } from "../ui/sheet";
import { Menu } from "lucide-react";
import { DialogTitle } from "../ui/dialog";
import {VisuallyHidden} from "@radix-ui/react-visually-hidden"

const navLinks = [
  { name: "Dashboard", href: "/" },
  { name: "Upload", href: "/upload" },
  { name: "Users", href: "/users" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <nav className="w-full bg-white border-b shadow-sm sticky top-0 z-50">
      <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-3">
        <Link href={"/"} className="font-bold text-xl">
          Contextual Bandit Admin
        </Link>
        <div className="hidden md:flex gap-4">
          {navLinks.map((link) => (
            <Link key={link.href} href={link.href}>
              <Button
                variant={pathname === link.href ? "default" : "ghost"}
                className={cn(
                  "capitalize",
                  pathname === link.href && "font-semibold"
                )}
              >
                {link.name}
              </Button>
            </Link>
          ))}
        </div>

        <div className="md:hidden">
          <Sheet open={open} onOpenChange={setOpen}>
            <SheetTrigger asChild>
              <Button variant={"ghost"} size={"icon"}>
                <Menu className="h-6 w-6" />
              </Button>
            </SheetTrigger>

            <SheetContent side="left" className="w-64">
                <VisuallyHidden>
                    <DialogTitle>Navigation Menu</DialogTitle>
                </VisuallyHidden>
              <nav className="flex flex-col gap-4 mt-6 ml-5">
                {navLinks.map((link) => (
                  <Link
                    key={link.href}
                    href={link.href}
                    onClick={() => setOpen(false)}
                    className={cn(
                      "text-lg",
                      pathname === link.href
                        ? "font-semibold text-blue-600"
                        : "text-gray-700"
                    )}
                  >
                    {link.name}
                  </Link>
                ))}
              </nav>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </nav>
  );
}
