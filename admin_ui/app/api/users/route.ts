import { NextResponse } from "next/server";
import { ddbDocClient } from "@/lib/dynamodb";
import { PutCommand, ScanCommand } from "@aws-sdk/lib-dynamodb";
import { v4 as uuidv4 } from "uuid";

export async function GET() {
  try {
    const data = await ddbDocClient.send(
      new ScanCommand({
        TableName: "Users",
        Limit: 5,
      })
    );

    return NextResponse.json({
      users: data.Items || [],
    });
  } catch (error) {
    console.error("Users fetching failed:", error);
    return NextResponse.json(
      {
        error: "Failed to fetch users",
      },
      {
        status: 500,
      }
    );
  }
}

export async function POST(req: Request) {
  try {
    const { name, age, location, interests } = await req.json();

    const newUser = {
      userId: uuidv4(),
      name,
      age,
      location,
      interests,
      createdAt: new Date().toISOString(),
    };

    await ddbDocClient.send(
      new PutCommand({
        TableName: "Users",
        Item: newUser,
      })
    );

    return NextResponse.json({ message: "User added successfully!" });
  } catch (error) {
    return NextResponse.json({ error: "Failed to add user!" }, { status: 500 });
  }
}
