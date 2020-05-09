import React, { useState } from 'react';

import Axios from 'axios';

import { Button, Typography, withStyles } from '@material-ui/core';

const styles = {
	overlay: {
		width: '100%',
		height: '100%',
		background: '#eeeeee',
		display: 'flex',
		justifyContent: 'center',
		alignItems: 'center',
	},
	popup: {
		background: 'white',
		display: 'flex',
		flexDirection: 'column',
		padding: '1rem',
		width: '400px',
		alignItems: 'center',
		borderRadius: '20px',
		boxShadow: '2px 2px 2px 2px #cccccc',
	},
	innerDiv: {
		display: 'flex',
		justifyContent: 'space-between',
		width: '100%',
		marginBottom: '5px',
	},
	popupText: {
		margin: 0,
		fontSize: '1.5rem',
	},
	popupLabel: {
		textAlign: 'center',
		fontSize: '1.5rem',
	},
	description: {
		textAlign: 'center',
		margin: 0,
	},
};

const Updater = ({ classes, features, restart, samples, setError }) => {
	const [currentSample, setCurrentSample] = useState(0);
	const [updating, setUpdating] = useState(false);

	const updateModel = (val) => {
		setUpdating(true);
		const sampleFeatures = {};
		features.forEach((f, i) => {
			if (i === 0) sampleFeatures[f] = Math.max(17, Math.round(samples[currentSample][i]));
			else if (i === 5) sampleFeatures[f] = Math.max(0, Math.round(samples[currentSample][5]));
			else sampleFeatures[f] = samples[currentSample][i];
		});
		Axios.get('/api/datasets/update_model/', {
			params: {
				sample: sampleFeatures,
				prediction: val,
			},
		})
			.then(() => {
				if (currentSample + 1 === samples.length) restart();
				setCurrentSample(currentSample + 1);
			})
			.catch(() =>
				setError('Something went wrong while updating. Please try again or come back later.'),
			)
			.finally(() => setUpdating(false));
	};

	return (
		<div style={styles.overlay}>
			<div style={styles.popup}>
				<h1>Model updater</h1>
				<p style={styles.description}>
					In order to properly update the model we will show you some samples closely related to the
					previously classified sample. Please indicate as what this sample should be classified so
					we can update the model accordingly.
				</p>
				<h2>
					Sample {currentSample + 1} / {samples.length}
				</h2>
				<div style={styles.innerDiv}>
					<label style={styles.popupLabel}>Age</label>
					<p style={styles.popupText}>{Math.max(17, Math.round(samples[currentSample][0]))}</p>
				</div>
				<div style={styles.innerDiv}>
					<label style={styles.popupLabel}>Occupation</label>
					<p style={styles.popupText}>{samples[currentSample][1]}</p>
				</div>
				<div style={styles.innerDiv}>
					<label style={styles.popupLabel}>Relationship</label>
					<p style={styles.popupText}>{samples[currentSample][2]}</p>
				</div>
				<div style={styles.innerDiv}>
					<label style={styles.popupLabel}>Race</label>
					<p style={styles.popupText}>{samples[currentSample][3]}</p>
				</div>
				<div style={styles.innerDiv}>
					<label style={styles.popupLabel}>Sex</label>
					<p style={styles.popupText}>{samples[currentSample][4]}</p>
				</div>
				<div style={styles.innerDiv}>
					<label style={styles.popupLabel}>Hours per week</label>
					<p style={styles.popupText}>{Math.max(0, Math.round(samples[currentSample][5]))}</p>
				</div>
				<div style={styles.innerDiv}>
					<label style={styles.popupLabel}>Native country</label>
					<p style={styles.popupText}>{samples[currentSample][6]}</p>
				</div>
				<div className={classes.controls}>
					<Button
						className={classes.button}
						variant="contained"
						color="secondary"
						onClick={() => updateModel('≤50K')}
						disabled={updating}
					>
						<Typography>{'≤50K'}</Typography>
					</Button>
					<Button
						className={classes.button}
						variant="contained"
						color="secondary"
						onClick={() => updateModel('>50K')}
						disabled={updating}
					>
						<Typography>{'>50K'}</Typography>
					</Button>
				</div>
			</div>
		</div>
	);
};

export default withStyles(styles)(Updater);
