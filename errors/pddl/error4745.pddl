(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b6 t0)
		(in-tower b6)
		(on b4 b6)
		(in-tower b4)
		(red b0)
		(blue b2)
		(blue b3)
		(red b4)
		(blue b4)
		(yellow b5)
		(green b7)
		(red b8)
		(green b9)
		(yellow b9)
		(red b3)
		(blue b1)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (not (on b8 b4))))
)