import React, { useState, useEffect } from 'react';

import Axios from 'axios';

import { makeStyles } from '@material-ui/core/styles';
import { AppBar, Button, Snackbar, Toolbar, Typography } from '@material-ui/core';
import Alert from '@material-ui/lab/Alert';

import Adult from './components/Adult';
import Updater from './components/Updater';

const useStyles = makeStyles((theme) => ({
	root: {
		flexGrow: 1,
		display: 'flex',
		flexDirection: 'column',
		height: '100vh',
	},
	menuButton: {
		marginRight: theme.spacing(2),
	},
	title: {
		flexGrow: 1,
	},
	button: {
		width: '10rem',
		height: '3rem',
		padding: '1rem',
		borderRadius: '1rem',
		margin: '1rem',
	},
	controls: {
		display: 'flex',
		flexDirection: 'row',
		width: '30vw',
		alignItems: 'center',
		justifyContent: 'space-evenly',
	},
}));

const App = () => {
	const classes = useStyles();
	const [error, setError] = useState(undefined);
	const [samples, setSamples] = useState([]);
	const [features, setFeatures] = useState([]);
	const [valuesPerFeature, setValuesPerFeature] = useState({});
	const [boundsPerFeature, setBoundsPerFeature] = useState({});

	useEffect(() => {
		Axios.get('/api/datasets/retrieve_adult/')
			.then((details) => {
				setFeatures(details.data.features);
				setValuesPerFeature(details.data.values_per_category);
				setBoundsPerFeature(details.data.bounds_per_feature);
			})
			.catch((_) =>
				setError(
					'Something went wrong while fetching the dataset. Please try another dataset or come back later.',
				),
			);
	}, []);

	return (
		<div className={classes.root}>
			<AppBar position="static">
				<Toolbar>
					<Typography variant="h6" className={classes.title}>
						Constrastive explanations
					</Typography>
					{samples.length > 0 && (
						<Button onClick={() => setSamples([])} color="inherit">
							Go back
						</Button>
					)}
				</Toolbar>
			</AppBar>
			{samples.length > 0 ? (
				<Updater
					restart={() => setSamples([])}
					features={features}
					classes={classes}
					setError={setError}
					samples={samples}
				/>
			) : (
				<Adult
					features={features}
					valuesPerFeature={valuesPerFeature}
					boundsPerFeature={boundsPerFeature}
					classes={classes}
					setError={setError}
					setSamples={setSamples}
				/>
			)}
			<Snackbar
				open={error !== undefined}
				autoHideDuration={6000}
				onClose={() => setError(undefined)}
			>
				<Alert onClose={() => setError(undefined)} severity="error">
					{error}
				</Alert>
			</Snackbar>
		</div>
	);
};

export default App;
