import React, { useState } from 'react';

import Main from './components/Main';

import { makeStyles } from '@material-ui/core/styles';
import { AppBar, Button, Toolbar, Typography } from '@material-ui/core';

const useStyles = makeStyles(theme => ({
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
}));

const App = () => {
	const classes = useStyles();
	const [datasetId, setDatasetId] = useState(54);

	return (
		<div className={classes.root}>
			<AppBar position="static">
				<Toolbar>
					<Typography variant="h6" className={classes.title}>
						Constrastive explanations
					</Typography>
					<Button onClick={() => setDatasetId(1590)} color="inherit">
						Adult
					</Button>
					<Button onClick={() => setDatasetId(31)} color="inherit">
						Credit-g
					</Button>
					<Button onClick={() => setDatasetId(54)} color="inherit">
						Vehicle
					</Button>
				</Toolbar>
			</AppBar>
			<Main datasetId={datasetId} />
		</div>
	);
};

export default App;
