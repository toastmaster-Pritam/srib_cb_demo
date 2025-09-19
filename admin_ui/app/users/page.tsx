"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { format } from "date-fns";
import React, { useEffect, useState } from "react";
import { toast } from "sonner";
enum DeviceType {
  MOBILE = "mobile",
  TABLET = "tablet",
  LAPTOP = "laptop",
}
interface User {
  userId: string;
  name: string;
  age: number;
  location: string;
  deviceType: DeviceType;
  interests: string[];
  createdAt: string;
}

const devices = Object.values(DeviceType);
export default function UserPage() {
  const [userData, setUserData] = useState({
    name: "",
    age: "",
    location: "",
    interests: [] as string[],
    deviceType: "" as DeviceType,
  });

  const [users, setUsers] = useState<User[]>([]);

  const fetchUsers = async () => {
    try {
      const res = await fetch("/api/users");
      if (!res.ok) throw new Error("Failed to fetch users!");
      const data = await res.json();
      setUsers(data.users);
      console.log("user",data)
    } catch (error) {
      toast.error("Error loading users");
    }
  };

    useEffect(() => {
      
      fetchUsers();
    }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setUserData((prev) => ({ ...prev, [name]: value }));
  };

  const handleAddInterest = (interest: string) => {
    setUserData((prev) => {
      if (prev.interests.includes(interest)) return prev;
      return { ...prev, interests: [...prev.interests, interest] };
    });
  };

  const handleRemoveInterest = (interest: string) => {
    setUserData((prev) => ({
      ...prev,
      interests: prev.interests.filter((i) => i !== interest),
    }));
  };

  const handleAddUser = async () => {
    if (
      !userData.name ||
      !userData.age ||
      !userData.deviceType ||
      !userData.location
    ) {
      toast.error("Please fill all required fields");
      return;
    }

    try {
      const res = await fetch("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...userData,
          age: Number(userData.age),
        }),
      });

      if (res.ok) {
        toast.success("User added successfully!", {
          description: `${format(
            new Date(),
            "EEEE, MMMM dd, yyyy 'at' h:mm a"
          )}`,
        });

        setUserData({
          name: "",
          age: "",
          location: "",
          interests: [],
          deviceType: "" as DeviceType,
        });

        fetchUsers();
      } else {
        toast.error("Failed to add user");
      }
    } catch (error) {
      toast.error("Something went wrong!");
    }
  };

  const Interests = ["Sports", "Music", "Technology", "Travel", "Food"];

  return (
    <div className="max-w-3xl mx-auto py-10 px-4">
      <h1 className="text-2xl font-bold mb-6">User Management</h1>

      <Card className="mb-8">
        <CardContent className="p-4 space-y-4">
          <Input
            name="name"
            placeholder="Name"
            value={userData.name}
            onChange={handleChange}
          />
          <Input
            name="age"
            placeholder="Age"
            value={userData.age}
            onChange={handleChange}
          />
          <Input
            name="location"
            placeholder="Location"
            value={userData.location}
            onChange={handleChange}
          />

          <div>
            <Select onValueChange={handleAddInterest} key={userData.interests.length}>
              <SelectTrigger>
                <SelectValue placeholder="Select an interest" />
              </SelectTrigger>
              <SelectContent>
                {Interests.map((interest, index) => (
                  <SelectItem value={interest.toLowerCase()} key={index}>
                    {interest}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {userData.interests.length > 0 && (
              <div className="flex gap-2 mt-2 flex-wrap">
                {userData.interests.map((interest) => (
                  <span
                    key={interest}
                    onClick={() => handleRemoveInterest(interest)}
                    className="bg-gray-200 px-2 py-1 rounded text-sm cursor-pointer hover:bg-gray-300"
                  >
                    {interest} ❌
                  </span>
                ))}
              </div>
            )}
          </div>

          <Select
            value={userData.deviceType}
            onValueChange={(value: DeviceType) =>
              setUserData((prev) => ({ ...prev, deviceType: value }))
            }
          >
            <SelectTrigger>
              <SelectValue placeholder="Select Device Type" />
            </SelectTrigger>
            <SelectContent>
              {devices.map((device, index) => (
                <SelectItem value={device} key={index}>
                  {device}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button onClick={handleAddUser}> Add User</Button>
        </CardContent>
      </Card>

      {/* Recent Users */}

      <h2 className="text-xl font-semibold mb-4">Recent Users</h2>
      <div className="space-y-4">
        {users.length > 0 ? (
          users.map((user) => (
            <Card key={user.userId}>
              <CardContent className="p-4">
                <p>
                  <strong>{user.name}</strong> ({user.age})
                </p>
                <p>Location: {user.location}</p>
                <p>Device Type: {user.deviceType}</p>
                <p>
                  Interests:
                  {user.interests.length > 0
                    ? user.interests.join(", ")
                    : "N/A"}
                </p>
                <p className="text-sm text-gray-500">
                  {" "}
                  Added:{" "}
                  {format(
                    new Date(user.createdAt),
                    "EEEE, MMMM dd,yyyy 'at' h:mm a"
                  )}
                </p>
              </CardContent>
            </Card>
          ))
        ) : (
          <p className="text-sm text-gray-500">No users yet.</p>
        )}
      </div>
    </div>
  );
}
