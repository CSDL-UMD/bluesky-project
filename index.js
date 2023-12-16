const express = require("express");
const mysql = require("mysql");
const pkg = require("@atproto/api");
const fs = require('fs');
const zlib = require('zlib');
const cbor = require('cbor');
const { Pool } = require('pg');
const { spawn } = require('child_process');
const BskyAgent = pkg.BskyAgent;
const extractUrls = require("extract-urls");
const { Subscription } = require('@atproto/xrpc-server');
const { cborToLexRecord, readCar } = require('@atproto/repo');
const agent = new BskyAgent({ service: "https://bsky.social"});
const app = express();
const fetch = require("node-fetch");
const parse = require('papaparse');
const subscription = new Subscription({
    service: `wss://bsky.network`,
    method: `com.atproto.sync.subscribeRepos`,
    getState: () => ({}),
    validate: (value) => value,
});
const pythonScript = './news_guard.py';
const currentDate = new Date();
const formattedDate = currentDate.toISOString().split('T')[0];
currentFileName = `./Messages/${formattedDate}.txt`;
let dfCombined;
let parsedAllSources;
let parsedMetadata;
let totalMessagesText = 0;
let below60Count = 0;
let above60Count = 0;
let totalMessages = 0;
let totalLinks = 0;
let intervalId;

require("dotenv").config();
app.set("views", "./pages");
app.set("view engine", "ejs");
app.use(express.static("public"));
app.use(express.static(__dirname + "/public"));
process.stdin.setEncoding("utf8");
global.fetch = fetch;

// Create connection
const db = new Pool({
	host: process.env.DB_HOST,
  	user: process.env.DB_USER,
  	password: process.env.DB_PASSWORD,
  	database: "bluesky_db",
  	port: 5432,
    charset: "utf8mb4"
});

db.connect((err) => {
  	if (err) {
		console.error("Error connecting to MySQL:", err);
  	}
	if (!process.env.BLUESKY_USERNAME || !process.env.BLUESKY_PASSWORD) {
        	console.error("Env variables are not defined!");
       		process.exit(1);
   	 }
});

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/app.html');
});

app.get('/get_data', (req, res) => {
    const query = 'SELECT day, totalmessages, totallinks, newsgreaterthan60, newslessthan60 FROM bsky_news';
    db.query(query, (err, results) => {
      if (err) {
        console.error('Error executing query:', err);
        res.status(500).json({ error: 'Internal Server Error' });
        return;
      }
      res.json(results);
    });
});

app.use(express.static('public'));
app.listen("3000", () => {});

// Define a function to promisify the database query
function queryPromise(sql, successMessage) {
    return new Promise((resolve, reject) => {
        db.query(sql, (err, result) => {
            if (err) {
                reject(err);
            } else {
                console.log(successMessage);
                resolve(result);
            }
        });
    });
}

const compressFile = async (fileName) => {
    const readStream = fs.createReadStream(fileName);
    const gzip = zlib.createGzip();
    const compressedFileName = fileName.replace(/\.[^.]+$/, '') + '.gz';
    const writeStream = fs.createWriteStream(compressedFileName);

    await new Promise((resolve, reject) => {
        readStream.pipe(gzip).pipe(writeStream);
        Promise.all([
            new Promise((resolve) => writeStream.on('close', resolve)),
            new Promise((resolve) => gzip.on('end', resolve)),
        ])
            .then(async () => {
                await fs.promises.unlink(fileName);
                resolve();
            })
            .catch(reject);

        writeStream.on('error', reject);
        gzip.on('error', reject);
    });
};

setInterval(async () => {
	const currentFileToCompress = currentFileName;
        const currentDate = new Date();
        const formattedDate = currentDate.toISOString().split('T')[0];
        currentFileName = `./Messages/${formattedDate}.txt`;
        await compressFile(currentFileToCompress);
	totalMessagesText = 0;
}, 86400000);

const initializeData = async () => {
    const metadataData = fs.readFileSync('./NewsGuard/metadata.csv', 'utf8');
    const allSourcesData = fs.readFileSync('./NewsGuard/all-sources-metadata.csv', 'utf8');

    try {
        [parsedMetadata, parsedAllSources] = await Promise.all([parseCSV(metadataData), parseCSV(allSourcesData)]);
        const commonColumns = getCommonColumns(parsedMetadata, parsedAllSources);
        const rowsToAppend = getRowsToAppend(parsedMetadata, parsedAllSources);
        dfCombined = concat(parsedAllSources, rowsToAppend, commonColumns);
    } catch (err) {
        console.error('Error:', err);
    }
}

const openFirehose = async (cborData) => {
    try {
        // Extract and log URLs from the post
        const car = await readCar(Uint8Array.from(cborData.blocks)); 
        for (const operation of cborData.ops) {
            if (operation.action !== 'create') {
                continue;
            }
    
            const recordBytes = car.blocks.get(operation.cid);
            if (!recordBytes) {
                continue;
            }
    
            const lexRecord = cborToLexRecord(recordBytes);
            if (lexRecord.text !== undefined && operation.path.split('/')[0] !== "app.bsky.feed.post") {
                continue;
            }
            if (lexRecord.text !== undefined){
                totalMessages++;
		fs.appendFileSync(currentFileName, JSON.stringify({ text: lexRecord.text }) + '\n');
            }
    
            if (lexRecord.text.toLowerCase().includes("https://")) {
                const urls = extractUrls(lexRecord.text);  
                for (const shortURL of urls) {
                    totalLinks++; 
                    processUrl(shortURL);
                }
            }
        }
    } catch (error) {
        // console.error(`Error: ${error.message}`);
    } 
}

async function findMatchingRow(longUrl) {
    if (!longUrl) {
        return null;
    }

    for (const row of parsedAllSources) {
        const source = row && row.Source;
        if (source && longUrl.toLowerCase().includes(source.toLowerCase())) {
            return row.Score; 
        }
    }
    return null;
}

async function unshortenUrlPython(shortUrl) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', ['news_guard.py', shortUrl]);

        let longUrl = '';
        let domain = '';

        pythonProcess.stdout.on('data', (data) => {
            const lines = data.toString().trim().split('\n');
            if (lines.length >= 2) {
                longUrl = lines[0].replace('Long URL: ', '');
                domain = lines[1].replace('Domain: ', '');
            }
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                resolve({ longUrl, domain });
            } else {
                reject(new Error(`Python script exited with code ${code}`));
            }
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Error: ${data}`);
            reject(new Error(data.toString()));
        });
    });
}

async function processUrl(shortUrl) {
    try {
	let score = null;
        const { longUrl, domain } = await unshortenUrlPython(shortUrl);
        const cleanedDomain = await replaceWWW(domain);
        try {
            const domainRow = dfCombined.find((row) => row && row.Domain === cleanedDomain);
            if (domainRow) {
                score = domainRow.Score;
            } else {
                score = await findMatchingRow(longUrl);
            }
        } catch (error) {
            console.error('Error:', error);
        }
        if (score && !isNaN(score)) {
            if (score < 60) {
                below60Count++;
            } else {
                above60Count++;
            }
        }
    } catch (error) {
        console.error('Error processing URL:', error);
    }
}

async function replaceWWW(input) {
    return input.replace('www.', '');
}


//MAIN METHOD
(async () => {
	try {
        await agent.login({
            identifier: process.env.BLUESKY_USERNAME,
            password: process.env.BLUESKY_PASSWORD,
        });
	await initializeData();
    } catch (error) {
        console.error(`Error: Failed to login. Please check your BLUESKY_USERNAME and BLUESKY_PASSWORD.`);
		throw(error);
    }
	eventHandler();
})();

const eventHandler = async () => {
    const insertDataIntoDatabase = async () => {
        const date = new Date();
        const currentDate = date.toISOString().split('.')[0].replace('T', ' ');
        try {
            const createDatabaseQuery = `
            INSERT INTO bsky_news (day, totalmessages, totallinks, newsgreaterthan60, newslessthan60)
            VALUES (
              '${currentDate}', 
              ${totalMessages}, 
              ${totalLinks}, 
              ${above60Count}, 
              ${below60Count}
            );
          `;
            await queryPromise(createDatabaseQuery, "Data inserted into the database successfully.");
        } catch (err) {
            console.error('Error inserting data into the database:', err);
        }

        below60Count = 0;
        above60Count = 0;
        totalLinks = 0;
        totalMessages = 0;
    };

    // Clear the previous interval and set a new one
    clearInterval(intervalId);
    intervalId = setInterval(insertDataIntoDatabase, 3600000);
    
    try {
        for await (const event of subscription) {
            openFirehose(event);
        }
    } catch (error) {
        // console.error(`Error in eventHandler: ${error.message}`);
    }
};

function parseCSV(data) {
    return new Promise((resolve, reject) => {
        parse.parse(data, {
            header: true,
            skipEmptyLines: true,
            complete: (result) => {
                const filteredData = result.data.map(row => ({
                    Score: row.Score,
                    Domain: row.Domain,
                    Source: row.Source
                }));
                resolve(filteredData);
            },
            error: (error) => reject(error)
        });
    });
}

function getCommonColumns(df, dfAll) {
    const dfColumns = Object.keys(df[0]);
    const dfAllColumns = Object.keys(dfAll[0]);
    return dfColumns.filter(col => dfAllColumns.includes(col));
}

function getRowsToAppend(df, dfAll) {
    const domains = dfAll.map(row => row.Domain);
    return df.filter(row => !domains.includes(row.Domain));
}

function concat(dfAll, rowsToAppend, commonColumns) {
    const dfCombined = [...dfAll, ...rowsToAppend];
    return dfCombined.map(row => {
        const newRow = {};
        commonColumns.forEach(col => newRow[col] = row[col]);
        return newRow;
    });
}
